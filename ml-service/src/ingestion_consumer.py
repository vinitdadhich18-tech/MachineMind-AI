"""
ingestion_consumer.py - MachineMind MQTT Telemetry Ingestion Consumer Service.

Phase 2: MQTT Consumer / Ingestion Service
Target Architecture Path:
NASA IMS -> Replay Simulator -> HiveMQ Cloud (TLS) -> MachineMind Ingestion Consumer -> TelemetryRecord Boundary

Key Architectural Boundaries:
1. NO dependency on Flask, MongoDB, or Streamlit on the live path.
2. Reuses telemetry_schema.py and mqtt_config.py from Phase 1.
3. Implements strict payload validation, label leakage assertions, per-machine sequence tracking,
   idempotent duplicate detection, and clean TelemetryRecord processing boundary.
4. Credentials loaded strictly from environment. Zero credentials logged.
"""

import sys
import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import numpy as np

# Ensure ml-service root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_SERVICE_DIR = PROJECT_ROOT / "ml-service"
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from src.mqtt_config import MQTTConfig, create_mqtt_client
from src.telemetry_schema import (
    validate_telemetry_payload,
    assert_no_label_leakage,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_CHANNELS,
)

logger = logging.getLogger("MachineMind.IngestionConsumer")


@dataclass
class TelemetryRecord:
    """
    Clean internal representation of validated vibration telemetry snapshot.
    Establishes the boundary between MQTT ingestion and downstream ML/storage pipelines.
    """
    machine_id: str
    snapshot_sequence: int
    original_timestamp: str
    ingest_timestamp: str
    sampling_rate_hz: float
    sample_count: int
    source: str
    channels: Dict[str, np.ndarray]  # Dict mapping 'ch1'..'ch4' -> 1D float64 numpy array (20480,)
    ingestion_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Returns metadata summary without serializing raw float arrays."""
        return {
            "machine_id": self.machine_id,
            "snapshot_sequence": self.snapshot_sequence,
            "original_timestamp": self.original_timestamp,
            "ingest_timestamp": self.ingest_timestamp,
            "sampling_rate_hz": self.sampling_rate_hz,
            "sample_count": self.sample_count,
            "source": self.source,
            "channels_present": list(self.channels.keys()),
            "ingestion_metadata": self.ingestion_metadata,
        }


@dataclass
class ConsumerMetrics:
    """Operational metrics counter for observability."""
    messages_received: int = 0
    messages_accepted: int = 0
    messages_rejected: int = 0
    duplicates_detected: int = 0
    out_of_order_detected: int = 0
    status_events_received: int = 0
    leakage_violations_detected: int = 0
    reconnection_events: int = 0
    last_processed_sequence: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages_received": self.messages_received,
            "messages_accepted": self.messages_accepted,
            "messages_rejected": self.messages_rejected,
            "duplicates_detected": self.duplicates_detected,
            "out_of_order_detected": self.out_of_order_detected,
            "status_events_received": self.status_events_received,
            "leakage_violations_detected": self.leakage_violations_detected,
            "reconnection_events": self.reconnection_events,
            "last_processed_sequence": self.last_processed_sequence,
        }


class TelemetryIngestionConsumer:
    """
    MQTT Ingestion Consumer Service for HiveMQ Cloud.
    """

    def __init__(
        self,
        config: MQTTConfig,
        record_callback: Optional[Callable[[TelemetryRecord], None]] = None
    ):
        self.config = config
        self.record_callback = record_callback
        self.metrics = ConsumerMetrics()
        self.client = None
        self.is_connected = False
        
        # Per-machine sequence state for idempotency and ordering
        # machine_id -> last_sequence_int
        self._last_sequences: Dict[str, int] = {}
        
        # Buffer to keep track of processed sequence IDs for windowed duplicate detection
        # machine_id -> set of sequence ints
        self._seen_sequences: Dict[str, set] = {}

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self.is_connected = True
        self.metrics.reconnection_events += 1
        
        telemetry_topic = f"{self.config.topic_prefix}/telemetry/+"
        status_topic = f"{self.config.topic_prefix}/status/+"

        client.subscribe(telemetry_topic, qos=self.config.qos)
        client.subscribe(status_topic, qos=self.config.qos)
        
        logger.info(
            f"Consumer successfully connected to HiveMQ Cloud. "
            f"Subscribed to '{telemetry_topic}' and '{status_topic}' (QoS {self.config.qos})."
        )

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.is_connected = False
        logger.warning(f"Consumer disconnected from MQTT broker (reason_code={reason_code}). Reconnect backoff active.")

    def _on_message(self, client, userdata, msg):
        start_time = time.time()
        self.metrics.messages_received += 1
        topic = msg.topic
        payload_bytes = msg.payload

        try:
            # Handle status event topic
            if "/status/" in topic:
                self.metrics.status_events_received += 1
                status_dict = json.loads(payload_bytes.decode('utf-8'))
                logger.info(f"[STATUS EVENT] Topic: '{topic}' payload: {status_dict}")
                return

            # Process Telemetry Topic
            payload_str = payload_bytes.decode('utf-8')
            payload_dict = json.loads(payload_str)

            # 1. Label Leakage Guard Assertion
            try:
                assert_no_label_leakage(payload_dict)
            except ValueError as err:
                self.metrics.messages_rejected += 1
                self.metrics.leakage_violations_detected += 1
                logger.error(f"[REJECTED - LABEL LEAKAGE] Topic '{topic}': {err}")
                return

            # 2. Schema Validation
            try:
                validate_telemetry_payload(payload_dict)
            except Exception as err:
                self.metrics.messages_rejected += 1
                logger.error(f"[REJECTED - MALFORMED SCHEMA] Topic '{topic}': {err}")
                return

            # 3. Extract Metadata & Check Sequence / Idempotency
            machine_id = payload_dict["machine_id"]
            seq = payload_dict["snapshot_sequence"]
            orig_ts = payload_dict["original_timestamp"]
            
            is_duplicate = False
            is_out_of_order = False
            is_gap = False

            if machine_id not in self._seen_sequences:
                self._seen_sequences[machine_id] = set()
                self._last_sequences[machine_id] = -1

            seen_set = self._seen_sequences[machine_id]
            last_seq = self._last_sequences[machine_id]

            # Idempotency check: QoS 1 retry handling
            if seq in seen_set:
                is_duplicate = True
                self.metrics.duplicates_detected += 1
                logger.warning(f"[IDEMPOTENCY DUPLICATE] Machine '{machine_id}' sequence {seq} already processed. Skipping duplicate.")
            else:
                seen_set.add(seq)

            # Sequence ordering check
            if last_seq >= 0:
                if seq < last_seq:
                    is_out_of_order = True
                    self.metrics.out_of_order_detected += 1
                    logger.warning(f"[OUT OF ORDER] Machine '{machine_id}' sequence {seq} arrived after {last_seq}.")
                elif seq > last_seq + 1:
                    is_gap = True
                    logger.warning(f"[SEQUENCE GAP DETECTED] Machine '{machine_id}' jumped from sequence {last_seq} to {seq}.")

            # Update last sequence tracker if monotonic
            if seq > last_seq:
                self._last_sequences[machine_id] = seq
                self.metrics.last_processed_sequence[machine_id] = seq

            # 4. Convert Raw Channel Lists to 1D Float64 Numpy Arrays
            # Avoid unneeded memory allocations by direct in-place numpy array view conversion
            channels_np: Dict[str, np.ndarray] = {}
            for ch_name in EXPECTED_CHANNELS:
                samples_list = payload_dict["channels"][ch_name]
                channels_np[ch_name] = np.asarray(samples_list, dtype=np.float64)

            processing_latency_ms = (time.time() - start_time) * 1000.0

            # 5. Build TelemetryRecord processing boundary object
            record = TelemetryRecord(
                machine_id=machine_id,
                snapshot_sequence=seq,
                original_timestamp=orig_ts,
                ingest_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                sampling_rate_hz=float(payload_dict["sampling_rate_hz"]),
                sample_count=int(payload_dict["sample_count"]),
                source=str(payload_dict["source"]),
                channels=channels_np,
                ingestion_metadata={
                    "is_duplicate": is_duplicate,
                    "is_out_of_order": is_out_of_order,
                    "is_sequence_gap": is_gap,
                    "processing_latency_ms": round(processing_latency_ms, 3),
                    "qos": msg.qos,
                    "retained": msg.retain,
                }
            )

            self.metrics.messages_accepted += 1
            logger.info(
                f"[ACCEPTED TELEMETRY] machine='{machine_id}' seq={seq} "
                f"orig_ts='{orig_ts}' latency={processing_latency_ms:.2f}ms "
                f"duplicate={is_duplicate} gap={is_gap}"
            )

            # 6. Invoke Downstream Callback if registered (and not duplicate)
            if self.record_callback and not is_duplicate:
                try:
                    self.record_callback(record)
                except Exception as cb_err:
                    logger.error(f"Error in record_callback for machine '{machine_id}': {cb_err}", exc_info=True)

        except json.JSONDecodeError as err:
            self.metrics.messages_rejected += 1
            logger.error(f"[REJECTED - INVALID JSON] Topic '{topic}': {err}")
        except Exception as err:
            self.metrics.messages_rejected += 1
            logger.error(f"[REJECTED - UNHANDLED EXCEPTION] Topic '{topic}': {err}", exc_info=True)

    def start(self) -> bool:
        """Starts MQTT Ingestion Consumer connection to HiveMQ Cloud over TLS."""
        if not self.config.is_configured():
            summary = self.config.get_masked_summary()
            logger.error(f"MQTT configuration incomplete. Masked summary: {summary}")
            return False

        client_id = f"machinemind-ingest-{self.config.machine_id}"
        self.client = create_mqtt_client(self.config, client_id=client_id, clean_session=False)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        summary = self.config.get_masked_summary()
        logger.info(f"Connecting Ingestion Consumer to HiveMQ Cloud host '{summary['broker_host']}' (Port {summary['broker_port']})...")

        try:
            self.client.connect(self.config.broker_host, self.config.broker_port, keepalive=60)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error(f"Ingestion Consumer failed to connect: {e}")
            return False

    def stop(self):
        """Stops MQTT Ingestion Consumer cleanly."""
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                logger.info("Ingestion Consumer cleanly disconnected.")
            except Exception as e:
                logger.warning(f"Error disconnecting Ingestion Consumer: {e}")
            finally:
                self.is_connected = False
