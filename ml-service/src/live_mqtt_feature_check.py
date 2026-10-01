import time

from src.mqtt_config import MQTTConfig
from src.ingestion_consumer import TelemetryIngestionConsumer, TelemetryRecord
from src.simulator import TelemetrySimulator
from src.canonical_feature_pipeline import CanonicalFeaturePipeline


def main():
    config = MQTTConfig()

    if not config.is_configured():
        print("ERROR: MQTT configuration is incomplete.")
        return

    pipeline = CanonicalFeaturePipeline()
    extracted_vectors = []

    def on_record(record: TelemetryRecord):
        print(
            f"[CONSUMER] Received seq={record.snapshot_sequence} "
            f"machine={record.machine_id}"
        )

        vector = pipeline.process_telemetry_record(record)
        vector.validate_finite()

        extracted_vectors.append(vector)

        print(
            f"[FEATURES] seq={record.snapshot_sequence} "
            f"shape={vector.feature_values.shape} "
            f"finite={vector.validate_finite()}"
        )

    consumer = TelemetryIngestionConsumer(
        config=config,
        record_callback=on_record
    )

    print("[1] Starting MQTT consumer...")
    if not consumer.start():
        print("ERROR: Consumer failed to connect.")
        return

    time.sleep(2)

    simulator = TelemetrySimulator(
        config=config,
        publish_interval=0.5
    )

    print("[2] Starting NASA IMS simulator...")
    if not simulator.start_connection():
        print("ERROR: Simulator failed to connect.")
        consumer.stop()
        return

    print("[3] Replaying 3 NASA IMS snapshots...")
    simulator.run_replay(
        start_idx=0,
        end_idx=3,
        loop=False
    )

    time.sleep(5)

    simulator.stop_connection()
    consumer.stop()

    print("\n========== VERIFICATION RESULT ==========")
    print(f"Telemetry records received: {len(extracted_vectors)}")

    for vector in extracted_vectors:
        print(
            f"seq={vector.snapshot_sequence} | "
            f"machine={vector.machine_id} | "
            f"features={vector.feature_values.shape} | "
            f"finite={vector.validate_finite()}"
        )

    if len(extracted_vectors) >= 3:
        print("\nSUCCESS: NASA IMS -> MQTT -> HiveMQ -> Consumer -> 36 features")
    else:
        print("\nFAILED: Expected 3 feature vectors.")


if __name__ == "__main__":
    main()