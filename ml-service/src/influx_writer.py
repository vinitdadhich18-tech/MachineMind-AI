"""
influx_writer.py - InfluxDB v2 Time-Series Writer Service.

Phase 3: InfluxDB Integration & Time-Series Schema
Target Dataset: NASA IMS Bearing Dataset (Set 2)

Architecture Path:
TelemetryRecord -> InfluxDBWriter -> InfluxDB v2 ('machinemind_telemetry' bucket)

Schema Rules:
1. Measurement 'vibration_features':
   - Tags: machine_id, channel, source (Low cardinality tags ONLY).
   - Fields: mean, std, rms, p2p, skewness, kurtosis, crest_factor, spectral_energy, spectral_centroid,
     original_timestamp (string), snapshot_sequence (integer).
2. Measurement 'pipeline_health':
   - Tags: service_name, machine_id.
   - Fields: snapshot_sequence, processing_latency_ms, is_duplicate, is_out_of_order, is_sequence_gap.
3. Waveform Rule: Raw 20480-element float arrays are NOT stored as individual fields. Derived summary features are stored.
4. Time Semantics: Ingest timestamp is point time; original_timestamp is retained as field for provenance.
"""

import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import influxdb_client
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

# Ensure ml-service root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_SERVICE_DIR = PROJECT_ROOT / "ml-service"
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from src.ingestion_consumer import TelemetryRecord
from src.feature_extraction import extract_time_features, extract_frequency_features

logger = logging.getLogger("MachineMind.InfluxWriter")

# Load environment variables
dotenv_paths = [PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"]
for p in dotenv_paths:
    if p.exists():
        load_dotenv(dotenv_path=p, override=False)


class InfluxConfig:
    """Configuration container for InfluxDB v2 connection."""

    def __init__(self, env_file: Optional[Path] = None):
        if env_file and env_file.exists():
            load_dotenv(dotenv_path=env_file, override=True)

        self.url: str = os.getenv("INFLUXDB_URL", "http://localhost:8086").strip()
        self.org: str = os.getenv("INFLUXDB_ORG", "machinemind_org").strip()
        self.bucket: str = os.getenv("INFLUXDB_BUCKET", "machinemind_telemetry").strip()
        self.token: str = os.getenv("INFLUXDB_TOKEN", "").strip()

    def is_configured(self) -> bool:
        """Checks if URL, org, bucket, and token are configured and non-placeholder."""
        is_valid_url = bool(self.url and "localhost" in self.url or "http" in self.url)
        is_valid_token = bool(self.token and not self.token.startswith("your_"))
        return is_valid_url and bool(self.org) and bool(self.bucket) and is_valid_token

    def get_masked_summary(self) -> Dict[str, Any]:
        """Returns connection parameters with masked authentication token."""
        masked_token = f"{self.token[:4]}***" if len(self.token) > 4 else "***"
        return {
            "url": self.url,
            "org": self.org,
            "bucket": self.bucket,
            "token": masked_token,
            "configured": self.is_configured(),
        }


class InfluxDBWriter:
    """
    InfluxDB v2 Writer for TelemetryRecord time-series data and health metrics.
    """

    def __init__(self, config: Optional[InfluxConfig] = None):
        self.config = config or InfluxConfig()
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api = None

    def connect(self) -> bool:
        """Establishes connection to InfluxDB v2 server."""
        if not self.config.is_configured():
            summary = self.config.get_masked_summary()
            logger.warning(f"InfluxDB config unconfigured or placeholder. Summary: {summary}")
            return False

        summary = self.config.get_masked_summary()
        logger.info(f"Connecting to InfluxDB at '{summary['url']}' (org='{summary['org']}', bucket='{summary['bucket']}')...")

        try:
            self.client = InfluxDBClient(
                url=self.config.url,
                token=self.config.token,
                org=self.config.org,
                timeout=10000,
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()

            # Health check ping
            health = self.client.health()
            if health.status == "pass":
                logger.info("Successfully connected to InfluxDB v2 server!")
                return True
            else:
                logger.warning(f"InfluxDB ping responded with status: {health.status}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB server at '{self.config.url}': {e}")
            self.client = None
            return False

    def close(self):
        """Closes InfluxDB client connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("InfluxDB connection closed.")
            except Exception as e:
                logger.warning(f"Error closing InfluxDB client: {e}")

    def write_telemetry_record(self, record: TelemetryRecord) -> bool:
        """
        Extracts summary features from TelemetryRecord and writes points to InfluxDB.

        Idempotency Rule: If record is marked duplicate (record.ingestion_metadata['is_duplicate'] == True),
        writing is skipped to prevent duplicate time-series points.
        """
        # Idempotency check: skip duplicate retries
        if record.ingestion_metadata.get("is_duplicate", False):
            logger.info(f"Skipping InfluxDB write for duplicate record machine='{record.machine_id}' seq={record.snapshot_sequence}.")
            return True

        if not self.write_api and not self.connect():
            logger.warning("InfluxDB write_api uninitialized. Skipping write.")
            return False

        try:
            # 1. Parse Ingest Timestamp as InfluxDB Point Time
            try:
                point_time = datetime.fromisoformat(record.ingest_timestamp.replace("Z", "+00:00"))
            except Exception:
                point_time = datetime.now(timezone.utc)

            # 2. Extract Canonical Feature Vector via CanonicalFeaturePipeline
            from src.canonical_feature_pipeline import CanonicalFeaturePipeline
            pipeline = CanonicalFeaturePipeline(sampling_rate_hz=record.sampling_rate_hz)
            vector = pipeline.process_telemetry_record(record)
            all_feats = vector.feature_dict

            points: List[Point] = []

            # 3. Create Points for Measurement 'vibration_features' (1 point per channel)
            for ch_idx in range(1, 5):
                ch_key = f"ch{ch_idx}"
                ch_name = f"Channel_{ch_idx}"

                point = (
                    Point("vibration_features")
                    .tag("machine_id", str(record.machine_id))
                    .tag("channel", ch_key)
                    .tag("source", str(record.source))
                    .field("snapshot_sequence", int(record.snapshot_sequence))
                    .field("original_timestamp", str(record.original_timestamp))
                    .field("mean", float(all_feats.get(f"{ch_key}_mean", 0.0)))
                    .field("std", float(all_feats.get(f"{ch_key}_std", 0.0)))
                    .field("rms", float(all_feats.get(f"{ch_key}_rms", 0.0)))
                    .field("p2p", float(all_feats.get(f"{ch_key}_p2p", 0.0)))
                    .field("skewness", float(all_feats.get(f"{ch_key}_skewness", 0.0)))
                    .field("kurtosis", float(all_feats.get(f"{ch_key}_kurtosis", 0.0)))
                    .field("crest_factor", float(all_feats.get(f"{ch_key}_crest_factor", 0.0)))
                    .field("spectral_energy", float(all_feats.get(f"{ch_key}_spectral_energy", 0.0)))
                    .field("spectral_centroid", float(all_feats.get(f"{ch_key}_spectral_centroid", 0.0)))
                    .time(point_time)
                )
                points.append(point)

            # 4. Create Point for Measurement 'pipeline_health'
            health_point = (
                Point("pipeline_health")
                .tag("service_name", "mqtt_ingestion_consumer")
                .tag("machine_id", str(record.machine_id))
                .field("snapshot_sequence", int(record.snapshot_sequence))
                .field("processing_latency_ms", float(record.ingestion_metadata.get("processing_latency_ms", 0.0)))
                .field("is_duplicate", int(record.ingestion_metadata.get("is_duplicate", False)))
                .field("is_out_of_order", int(record.ingestion_metadata.get("is_out_of_order", False)))
                .field("is_sequence_gap", int(record.ingestion_metadata.get("is_sequence_gap", False)))
                .time(point_time)
            )
            points.append(health_point)

            # 5. Execute Write API Call
            self.write_api.write(bucket=self.config.bucket, org=self.config.org, record=points)
            logger.info(
                f"[INFLUX WRITE SUCCESS] machine='{record.machine_id}' seq={record.snapshot_sequence} "
                f"points_written={len(points)} bucket='{self.config.bucket}'"
            )
            return True

        except Exception as err:
            logger.error(f"[INFLUX WRITE FAILURE] Failed writing record for machine '{record.machine_id}': {err}", exc_info=True)
            return False

    def query_telemetry_features(self, machine_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Queries stored vibration_features points from InfluxDB using Flux query."""
        if not self.query_api and not self.connect():
            return []

        query = f'''
        from(bucket: "{self.config.bucket}")
          |> range(start: -30d)
          |> filter(fn: (r) => r._measurement == "vibration_features")
          |> filter(fn: (r) => r.machine_id == "{machine_id}")
          |> limit(n: {limit})
        '''
        try:
            result = self.query_api.query(org=self.config.org, query=query)
            records = []
            for table in result:
                for r in table.records:
                    records.append({
                        "time": str(r.get_time()),
                        "measurement": r.get_measurement(),
                        "channel": r.values.get("channel"),
                        "field": r.get_field(),
                        "value": r.get_value(),
                        "snapshot_sequence": r.values.get("snapshot_sequence"),
                        "original_timestamp": r.values.get("original_timestamp"),
                    })
            return records
        except Exception as e:
            logger.error(f"Error executing Flux query for machine '{machine_id}': {e}")
            return []
