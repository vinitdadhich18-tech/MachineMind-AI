"""
simulator.py - NASA IMS Telemetry Replay Simulator / MQTT Publisher.

Phase 1: Telemetry Simulator & MQTT Infrastructure
Target Dataset: NASA IMS Bearing Dataset (Set 2)

Replays historical run-to-failure raw vibration snapshots in strict chronological order
and publishes them to HiveMQ Cloud over TLS.

Honesty Principle: Data is explicitly identified as 'replayed_nasa_ims'.
Security Principle: Credentials loaded safely from environment. Zero secrets logged.
"""

import sys
import time
import signal
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import (
    DEFAULT_RAW_SET2_PATH,
    get_snapshot_files,
    parse_snapshot_timestamp,
    load_snapshot,
)
from src.mqtt_config import MQTTConfig, create_mqtt_client
from src.telemetry_schema import (
    build_telemetry_payload,
    serialize_payload,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MachineMind.Simulator")


class TelemetrySimulator:
    """
    NASA IMS Chronological Telemetry Replay Engine.
    """

    def __init__(
        self,
        config: MQTTConfig,
        raw_dir: Path | None = None,
        publish_interval: float = 1.0
    ):
        self.config = config
        self.raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_SET2_PATH
        self.publish_interval = max(0.01, float(publish_interval))
        self.running = False
        self.client = None

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw IMS dataset directory not found at: {self.raw_dir}")

        self.snapshot_files = get_snapshot_files(self.raw_dir)
        if not self.snapshot_files:
            raise ValueError(f"No snapshot files found in: {self.raw_dir}")

        logger.info(
            f"Initialized TelemetrySimulator with {len(self.snapshot_files)} snapshots. "
            f"Cadence: {self.publish_interval}s per snapshot."
        )

    def _setup_client(self):
        """Creates Paho client with status Last Will and Testament."""
        client_id = f"machinemind-sim-{self.config.machine_id}"
        client = create_mqtt_client(self.config, client_id=client_id, clean_session=True)

        status_topic = self.config.get_status_topic()
        
        # Configure Last Will and Testament (LWT) for ungraceful disconnects
        lwt_payload = f'{{"machine_id": "{self.config.machine_id}", "status": "offline", "reason": "connection_lost"}}'
        client.will_set(status_topic, payload=lwt_payload, qos=1, retain=True)

        return client

    def start_connection(self) -> bool:
        """Connects to HiveMQ Cloud broker over TLS."""
        if not self.config.is_configured():
            masked = self.config.get_masked_summary()
            logger.error(
                f"MQTT configuration incomplete or missing. Summary: {masked}. "
                "Please configure .env with valid HiveMQ Cloud credentials."
            )
            return False

        summary = self.config.get_masked_summary()
        logger.info(f"Connecting to HiveMQ Cloud broker at {summary['broker_host']}:{summary['broker_port']} (TLS={summary['use_tls']})...")

        try:
            self.client = self._setup_client()
            self.client.connect(self.config.broker_host, self.config.broker_port, keepalive=60)
            self.client.loop_start()

            # Publish online status
            status_topic = self.config.get_status_topic()
            online_payload = f'{{"machine_id": "{self.config.machine_id}", "status": "online", "time": "{datetime.now(timezone.utc).isoformat()}"}}'
            self.client.publish(status_topic, online_payload, qos=1, retain=True)

            logger.info("Successfully connected to HiveMQ Cloud. Published 'online' status.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to HiveMQ Cloud broker: {e}")
            return False

    def stop_connection(self):
        """Gracefully disconnects and publishes offline status."""
        if self.client:
            try:
                status_topic = self.config.get_status_topic()
                offline_payload = f'{{"machine_id": "{self.config.machine_id}", "status": "offline", "reason": "graceful_shutdown"}}'
                self.client.publish(status_topic, offline_payload, qos=1, retain=True)
                time.sleep(0.2)
                self.client.loop_stop()
                self.client.disconnect()
                logger.info("Gracefully disconnected from HiveMQ Cloud broker.")
            except Exception as e:
                logger.warning(f"Error during MQTT disconnect: {e}")

    def run_replay(self, start_idx: int = 0, end_idx: int | None = None, loop: bool = False):
        """
        Runs chronological replay loop publishing snapshots over MQTT.
        """
        total_files = len(self.snapshot_files)
        target_end_idx = min(end_idx if end_idx is not None else total_files, total_files)
        
        self.running = True
        telemetry_topic = self.config.get_telemetry_topic()

        logger.info(
            f"Starting NASA IMS replay stream to topic '{telemetry_topic}' "
            f"(snapshots {start_idx} to {target_end_idx - 1} of {total_files})."
        )

        try:
            while self.running:
                for idx in range(start_idx, target_end_idx):
                    if not self.running:
                        break

                    fpath = self.snapshot_files[idx]
                    orig_dt = parse_snapshot_timestamp(fpath)
                    
                    # Load raw 20480x4 snapshot
                    df_raw = load_snapshot(fpath)

                    # Build telemetry JSON payload
                    payload_dict = build_telemetry_payload(
                        raw_snapshot=df_raw,
                        machine_id=self.config.machine_id,
                        snapshot_sequence=idx,
                        original_timestamp=orig_dt,
                        sampling_rate_hz=20480.0
                    )

                    json_str = serialize_payload(payload_dict)
                    payload_kb = len(json_str.encode('utf-8')) / 1024.0

                    # Publish via MQTT over TLS (QoS 1, retain=False)
                    if self.client:
                        info = self.client.publish(telemetry_topic, json_str, qos=self.config.qos, retain=False)
                        info.wait_for_publish(timeout=10.0)

                    logger.info(
                        f"[{idx + 1}/{total_files}] Published snapshot sequence={idx} "
                        f"file='{fpath.name}' orig_ts='{orig_dt}' size={payload_kb:.1f}KB"
                    )

                    time.sleep(self.publish_interval)

                if not loop:
                    logger.info("Replay completed (loop=False). Exiting replay loop.")
                    break
                else:
                    logger.info("Replay iteration finished. Restarting from start_idx (loop=True)...")

        except KeyboardInterrupt:
            logger.info("Replay interrupted by user.")
        finally:
            self.stop_connection()


def main():
    parser = argparse.ArgumentParser(description="MachineMind AI — NASA IMS Telemetry Replay Simulator")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting snapshot index (default: 0)")
    parser.add_argument("--end-idx", type=int, default=None, help="Ending snapshot index (default: all)")
    parser.add_argument("--interval", type=float, default=1.0, help="Publish cadence interval in seconds (default: 1.0)")
    parser.add_argument("--loop", action="store_true", help="Loop replay indefinitely")
    args = parser.parse_args()

    config = MQTTConfig()
    simulator = TelemetrySimulator(config=config, publish_interval=args.interval)

    # Handle termination signals cleanly
    def signal_handler(sig, frame):
        logger.info("Termination signal received. Shutting down simulator...")
        simulator.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if simulator.start_connection():
        simulator.run_replay(start_idx=args.start_idx, end_idx=args.end_idx, loop=args.loop)


if __name__ == "__main__":
    main()
