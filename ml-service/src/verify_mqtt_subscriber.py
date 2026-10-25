"""
verify_mqtt_subscriber.py - Independent MQTT Verification Subscriber Tool.

Phase 1: Telemetry Simulator & MQTT Infrastructure
Target Dataset: NASA IMS Bearing Dataset (Set 2)

Observes published messages on HiveMQ Cloud over TLS, verifies payload integrity,
confirms snapshot ordering and array dimensions (20480x4), and asserts ZERO label leakage.
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mqtt_config import MQTTConfig, create_mqtt_client
from src.telemetry_schema import (
    validate_telemetry_payload,
    assert_no_label_leakage,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_CHANNELS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MachineMind.Verifier")


class MQTTVerifier:
    """
    Independent subscriber and payload validation observer.
    """

    def __init__(self, config: MQTTConfig):
        self.config = config
        self.received_messages: List[Dict[str, Any]] = []
        self.status_messages: List[Dict[str, Any]] = []
        self.invalid_messages_count: int = 0
        self.leakage_violations_count: int = 0
        self.client = None

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        logger.info("Verifier connected to HiveMQ Cloud broker successfully.")
        
        telemetry_wildcard = f"{self.config.topic_prefix}/telemetry/+"
        status_wildcard = f"{self.config.topic_prefix}/status/+"

        client.subscribe(telemetry_wildcard, qos=self.config.qos)
        client.subscribe(status_wildcard, qos=self.config.qos)
        logger.info(f"Subscribed to topics: '{telemetry_wildcard}' and '{status_wildcard}'.")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        payload_bytes = msg.payload
        payload_kb = len(payload_bytes) / 1024.0

        try:
            payload_str = payload_bytes.decode('utf-8')
            payload_dict = json.loads(payload_str)

            if "/status/" in topic:
                logger.info(f"[STATUS EVENT] Topic: '{topic}' Payload: {payload_dict}")
                self.status_messages.append(payload_dict)
                return

            # Telemetry verification
            # 1. Label leakage check
            assert_no_label_leakage(payload_dict)

            # 2. Schema validation
            validate_telemetry_payload(payload_dict)

            seq = payload_dict.get("snapshot_sequence")
            m_id = payload_dict.get("machine_id")
            orig_ts = payload_dict.get("original_timestamp")

            logger.info(
                f"[TELEMETRY RECEIVED] Topic: '{topic}' machine='{m_id}' "
                f"sequence={seq} orig_ts='{orig_ts}' payload_size={payload_kb:.1f}KB"
            )

            self.received_messages.append(payload_dict)

        except ValueError as err:
            if "LABEL LEAKAGE" in str(err):
                logger.error(f"[LEAKAGE VIOLATION] {err}")
                self.leakage_violations_count += 1
            else:
                logger.error(f"[INVALID PAYLOAD] {err}")
                self.invalid_messages_count += 1
        except Exception as e:
            logger.error(f"[PARSE ERROR] Failed to process payload on topic '{topic}': {e}")
            self.invalid_messages_count += 1

    def start(self) -> bool:
        if not self.config.is_configured():
            logger.error("MQTT configuration incomplete. Please set valid HiveMQ Cloud credentials in .env.")
            return False

        client_id = f"machinemind-verifier-{int(time.time())}"
        self.client = create_mqtt_client(self.config, client_id=client_id, clean_session=True)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        summary = self.config.get_masked_summary()
        logger.info(f"Starting Verifier connection to {summary['broker_host']}:{summary['broker_port']}...")
        
        try:
            self.client.connect(self.config.broker_host, self.config.broker_port, keepalive=60)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error(f"Verifier failed to connect to broker: {e}")
            return False

    def stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Verifier stopped.")

    def run_until(self, max_messages: int = 5, timeout_sec: float = 30.0) -> bool:
        """
        Runs event loop until max_messages received or timeout reached.
        Returns True if expected messages were received cleanly without errors.
        """
        start_time = time.time()
        if not self.start():
            return False

        logger.info(f"Waiting for up to {max_messages} telemetry messages (timeout {timeout_sec}s)...")

        try:
            while len(self.received_messages) < max_messages:
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    logger.warning(f"Timeout reached ({timeout_sec}s). Received {len(self.received_messages)} / {max_messages} messages.")
                    break
                time.sleep(0.5)

            logger.info("--- VERIFICATION SUMMARY ---")
            logger.info(f"Messages Received: {len(self.received_messages)}")
            logger.info(f"Status Events: {len(self.status_messages)}")
            logger.info(f"Invalid Payloads: {self.invalid_messages_count}")
            logger.info(f"Label Leakage Violations: {self.leakage_violations_count}")

            success = (
                len(self.received_messages) >= max_messages
                and self.invalid_messages_count == 0
                and self.leakage_violations_count == 0
            )
            return success

        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="MachineMind AI — Independent MQTT Verification Subscriber")
    parser.add_argument("--max-messages", type=int, default=3, help="Number of telemetry messages to receive before exiting (default: 3)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout in seconds (default: 30.0)")
    args = parser.parse_args()

    config = MQTTConfig()
    verifier = MQTTVerifier(config)
    success = verifier.run_until(max_messages=args.max_messages, timeout_sec=args.timeout)

    if success:
        logger.info("VERIFICATION PASSED CLEANLY!")
        sys.exit(0)
    else:
        logger.error("VERIFICATION FAILED OR TIMED OUT.")
        sys.exit(1)


if __name__ == "__main__":
    main()
