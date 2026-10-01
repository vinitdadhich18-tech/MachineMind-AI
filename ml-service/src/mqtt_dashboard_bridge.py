"""
MachineMind AI - MQTT -> Streamlit Dashboard Bridge

Architecture:
NASA IMS historical telemetry
        ↓
MQTT Simulator
        ↓
HiveMQ Cloud
        ↓
TelemetryIngestionConsumer
        ↓
TelemetryRecord
        ↓
CanonicalFeaturePipeline
        ↓
frontend/data/live_telemetry.json
        ↓
Streamlit Dashboard
"""

import json
import sys
import time
from pathlib import Path
from threading import Lock


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_SERVICE_DIR = PROJECT_ROOT / "ml-service"
FRONTEND_DATA_DIR = PROJECT_ROOT / "frontend" / "data"
STATE_FILE = FRONTEND_DATA_DIR / "live_telemetry.json"


# Make ml-service importable
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))


# ============================================================
# MACHINE MIND IMPORTS
# ============================================================

from src.ingestion_consumer import TelemetryIngestionConsumer
from src.mqtt_config import MQTTConfig
from src.canonical_feature_pipeline import CanonicalFeaturePipeline


# ============================================================
# THREAD SAFETY
# ============================================================

state_lock = Lock()


# ============================================================
# WRITE LIVE STATE
# ============================================================

def write_state(record):
    """
    Receives a validated TelemetryRecord from the MQTT ingestion
    consumer, extracts the canonical 36-feature vector, and writes
    a lightweight JSON state file for the Streamlit dashboard.
    """

    try:
        # --------------------------------------------------------
        # Run the SAME canonical feature pipeline used by the
        # streaming ML path.
        # --------------------------------------------------------

        pipeline = CanonicalFeaturePipeline(
            sampling_rate_hz=record.sampling_rate_hz
        )

        vector = pipeline.process_telemetry_record(record)

        features = vector.feature_dict

        # --------------------------------------------------------
        # Convert 36 features into dashboard-friendly
        # channel-wise structure.
        # --------------------------------------------------------

        channels = []

        for ch in range(1, 5):

            prefix = f"ch{ch}_"

            channels.append(
                {
                    "channel": ch,

                    "mean": float(
                        features[f"{prefix}mean"]
                    ),

                    "std": float(
                        features[f"{prefix}std"]
                    ),

                    "rms": float(
                        features[f"{prefix}rms"]
                    ),

                    "p2p": float(
                        features[f"{prefix}p2p"]
                    ),

                    "skewness": float(
                        features[f"{prefix}skewness"]
                    ),

                    "kurtosis": float(
                        features[f"{prefix}kurtosis"]
                    ),

                    "crest_factor": float(
                        features[f"{prefix}crest_factor"]
                    ),

                    "spectral_energy": float(
                        features[f"{prefix}spectral_energy"]
                    ),

                    "spectral_centroid": float(
                        features[f"{prefix}spectral_centroid"]
                    ),
                }
            )

        # --------------------------------------------------------
        # Build dashboard state
        # --------------------------------------------------------

        state = {
            "machine_id": record.machine_id,

            "snapshot_sequence": int(
                record.snapshot_sequence
            ),

            "original_timestamp": record.original_timestamp,

            "ingest_timestamp": record.ingest_timestamp,

            "sampling_rate_hz": float(
                record.sampling_rate_hz
            ),

            "sample_count": int(
                record.sample_count
            ),

            "source": record.source,

            "feature_count": len(features),

            "channels": channels,

            "ingestion": {
                "is_duplicate": bool(
                    record.ingestion_metadata.get(
                        "is_duplicate",
                        False
                    )
                ),

                "is_out_of_order": bool(
                    record.ingestion_metadata.get(
                        "is_out_of_order",
                        False
                    )
                ),

                "is_sequence_gap": bool(
                    record.ingestion_metadata.get(
                        "is_sequence_gap",
                        False
                    )
                ),

                "processing_latency_ms": float(
                    record.ingestion_metadata.get(
                        "processing_latency_ms",
                        0
                    )
                ),
            },

            "dashboard_updated_at": time.time(),
        }

        # --------------------------------------------------------
        # Ensure frontend/data exists
        # --------------------------------------------------------

        FRONTEND_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------------
        # Atomic file write
        #
        # Write to temporary file first and then replace the
        # existing JSON file. This prevents Streamlit from reading
        # a partially-written JSON file.
        # --------------------------------------------------------

        temp_file = STATE_FILE.with_suffix(".tmp")

        with state_lock:

            temp_file.write_text(
                json.dumps(
                    state,
                    indent=2
                ),
                encoding="utf-8"
            )

            temp_file.replace(
                STATE_FILE
            )

        # --------------------------------------------------------
        # Console confirmation
        # --------------------------------------------------------

        latency = state["ingestion"][
            "processing_latency_ms"
        ]

        print(
            f"[DASHBOARD UPDATE] "
            f"machine={record.machine_id} "
            f"seq={record.snapshot_sequence} "
            f"features={len(features)} "
            f"latency={latency}ms"
        )

    except Exception as error:

        print(
            f"[DASHBOARD ERROR] "
            f"Failed to process telemetry: {error}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MachineMind AI - MQTT Dashboard Bridge")
    print("=" * 60)

    # --------------------------------------------------------
    # Create MQTT configuration
    # --------------------------------------------------------

    config = MQTTConfig()

    # --------------------------------------------------------
    # Create the existing MachineMind MQTT consumer.
    #
    # IMPORTANT:
    # TelemetryIngestionConsumer requires BOTH:
    #   1. config
    #   2. record_callback
    # --------------------------------------------------------

    consumer = TelemetryIngestionConsumer(
        config=config,
        record_callback=write_state
    )

    # --------------------------------------------------------
    # Start MQTT consumer
    # --------------------------------------------------------

    if not consumer.start():

        print(
            "ERROR: MQTT consumer could not start."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Connected
    # --------------------------------------------------------

    print()
    print("CONNECTED TO HIVEMQ CLOUD")
    print("Waiting for telemetry...")
    print()
    print(
        f"Dashboard state file:"
    )
    print(
        f"{STATE_FILE}"
    )
    print()

    # --------------------------------------------------------
    # Keep bridge alive
    # --------------------------------------------------------

    try:

        while True:

            time.sleep(1)

    except KeyboardInterrupt:

        print()
        print(
            "Stopping dashboard bridge..."
        )

    finally:

        consumer.stop()

        print(
            "Dashboard bridge stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()