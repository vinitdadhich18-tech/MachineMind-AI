import time

from src.mqtt_config import MQTTConfig
from src.ingestion_consumer import TelemetryIngestionConsumer, TelemetryRecord
from src.simulator import TelemetrySimulator
from src.canonical_feature_pipeline import CanonicalFeaturePipeline


EXPECTED_SNAPSHOTS = 3
EXPECTED_FEATURE_COUNT = 36


def main():
    config = MQTTConfig()

    print("=" * 60)
    print("MachineMind AI - MQTT Pipeline Verification")
    print("=" * 60)

    if not config.is_configured():
        print("ERROR: MQTT configuration is incomplete.")
        return

    summary = config.get_masked_summary()

    print(f"Broker       : {summary['broker_host']}")
    print(f"Port         : {summary['broker_port']}")
    print(f"TLS          : {summary['use_tls']}")
    print(f"QoS          : {summary['qos']}")
    print(f"Machine      : {summary['machine_id']}")
    print(f"Topic prefix : {summary['topic_prefix']}")

    pipeline = CanonicalFeaturePipeline()
    extracted_vectors = []

    def on_record(record: TelemetryRecord):
        print(
            f"[CONSUMER] seq={record.snapshot_sequence} "
            f"machine={record.machine_id} "
            f"duplicate={record.ingestion_metadata.get('is_duplicate')} "
            f"gap={record.ingestion_metadata.get('is_sequence_gap')} "
            f"latency={record.ingestion_metadata.get('processing_latency_ms')}ms"
        )

        vector = pipeline.process_telemetry_record(record)
        finite = vector.validate_finite()

        extracted_vectors.append(vector)

        print(
            f"[FEATURES] seq={record.snapshot_sequence} "
            f"shape={vector.feature_values.shape} "
            f"finite={finite}"
        )

    consumer = TelemetryIngestionConsumer(
        config=config,
        record_callback=on_record,
    )

    print("\n[1] Starting MQTT consumer...")

    if not consumer.start():
        print("ERROR: Consumer failed to connect.")
        return

    time.sleep(2)

    simulator = TelemetrySimulator(
        config=config,
        publish_interval=0.5,
    )

    print("[2] Starting NASA IMS simulator...")

    if not simulator.start_connection():
        print("ERROR: Simulator failed to connect.")
        consumer.stop()
        return

    print("[3] Replaying 3 NASA IMS snapshots...")

    simulator.run_replay(
        start_idx=0,
        end_idx=EXPECTED_SNAPSHOTS,
        loop=False,
    )

    # Allow MQTT callbacks and feature processing to complete.
    time.sleep(5)

    simulator.stop_connection()
    consumer.stop()

    metrics = consumer.metrics

    feature_count_ok = all(
        vector.feature_values.shape == (EXPECTED_FEATURE_COUNT,)
        for vector in extracted_vectors
    )

    finite_ok = all(
        vector.validate_finite()
        for vector in extracted_vectors
    )

    sequences = [
        vector.snapshot_sequence
        for vector in extracted_vectors
    ]

    expected_sequences = list(range(EXPECTED_SNAPSHOTS))
    sequence_ok = sequences == expected_sequences

    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)

    print(f"Telemetry received : {metrics.messages_received}")
    print(f"Telemetry accepted : {metrics.messages_accepted}")
    print(f"Telemetry rejected : {metrics.messages_rejected}")
    print(f"Feature vectors    : {len(extracted_vectors)}")
    print(f"Sequences          : {sequences}")
    print(f"Duplicates         : {metrics.duplicates_detected}")
    print(f"Out-of-order       : {metrics.out_of_order_detected}")
    print(f"Feature count      : {EXPECTED_FEATURE_COUNT}")
    print(f"All features finite: {finite_ok}")

    print("\nExpected:")
    print(f"  Snapshots         = {EXPECTED_SNAPSHOTS}")
    print(f"  Features/vector   = {EXPECTED_FEATURE_COUNT}")
    print("  Duplicates        = 0")
    print("  Out-of-order      = 0")
    print("  Sequence gap      = 0")

    success = (
        metrics.messages_received >= EXPECTED_SNAPSHOTS
        and metrics.messages_accepted >= EXPECTED_SNAPSHOTS
        and metrics.messages_rejected == 0
        and len(extracted_vectors) >= EXPECTED_SNAPSHOTS
        and metrics.duplicates_detected == 0
        and metrics.out_of_order_detected == 0
        and feature_count_ok
        and finite_ok
        and sequence_ok
    )

    print("\n" + "-" * 60)

    if success:
        print("STATUS: PASS")
        print(
            "NASA IMS -> MQTT -> HiveMQ Cloud -> "
            "Consumer -> TelemetryRecord -> 36 features"
        )
    else:
        print("STATUS: FAIL")
        print("One or more MQTT pipeline verification checks failed.")

    print("=" * 60)


if __name__ == "__main__":
    main()