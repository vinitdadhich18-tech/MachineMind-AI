"""
telemetry_service.py - Telemetry validation, sequence integrity, and persistence for MachineMind AI.
"""

import time
import math
import logging
from typing import Any, Dict, Optional
from flask import current_app
from pymongo.errors import DuplicateKeyError, PyMongoError

from backend.utils.db import get_db
from backend.utils.errors import APIError
from backend.utils.responses import get_utc_now_iso
from backend.services.data_service import validate_machine_id

logger = logging.getLogger(__name__)

REQUIRED_CHANNEL_METRICS = (
    "mean",
    "std",
    "rms",
    "p2p",
    "skewness",
    "kurtosis",
    "crest_factor",
    "spectral_energy",
    "spectral_centroid"
)


def get_database():
    """Helper to retrieve database instance or raise 503 if unavailable."""
    db = get_db(current_app.config["MONGO_URI"], current_app.config["DATABASE_NAME"])
    if db is None:
        raise APIError("DATABASE_UNAVAILABLE", "MongoDB database connection is unavailable.", status_code=503)
    return db


def validate_telemetry_payload(payload: Any) -> Dict[str, Any]:
    """
    Strictly validates incoming live telemetry payload structure and values.
    Raises APIError (HTTP 400) on any malformed or out-of-bounds fields.
    """
    if not isinstance(payload, dict):
        raise APIError("VALIDATION_ERROR", "Request payload must be a JSON object.", status_code=400)

    # 1. Machine ID validation
    raw_machine_id = payload.get("machine_id")
    clean_machine_id = validate_machine_id(raw_machine_id)

    # 2. Snapshot Sequence (non-negative integer)
    seq = payload.get("snapshot_sequence")
    if seq is None or isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise APIError(
            code="VALIDATION_ERROR",
            message="Field 'snapshot_sequence' is required and must be a non-negative integer.",
            status_code=400,
            details={"snapshot_sequence": seq}
        )

    # 3. Timestamps
    orig_ts = payload.get("original_timestamp")
    if not orig_ts or not isinstance(orig_ts, str) or not orig_ts.strip():
        raise APIError(
            code="VALIDATION_ERROR",
            message="Field 'original_timestamp' is required and must be a non-empty string.",
            status_code=400
        )

    ingest_ts = payload.get("ingest_timestamp")
    if not ingest_ts or not isinstance(ingest_ts, str) or not ingest_ts.strip():
        raise APIError(
            code="VALIDATION_ERROR",
            message="Field 'ingest_timestamp' is required and must be a non-empty string.",
            status_code=400
        )

    # 4. Channels (must contain exactly 4 channels for 4-channel vibration rig)
    channels = payload.get("channels")
    if not isinstance(channels, list) or len(channels) != 4:
        raise APIError(
            code="VALIDATION_ERROR",
            message="Field 'channels' must be a list containing exactly 4 channels.",
            status_code=400,
            details={"channels_count": len(channels) if isinstance(channels, list) else None}
        )

    seen_channels = set()
    cleaned_channels = []
    for ch_idx, ch_data in enumerate(channels):
        if not isinstance(ch_data, dict):
            raise APIError(
                code="VALIDATION_ERROR",
                message=f"Channel entry at index {ch_idx} must be a JSON object.",
                status_code=400
            )

        ch_num = ch_data.get("channel")
        if isinstance(ch_num, bool) or not isinstance(ch_num, int) or ch_num not in (1, 2, 3, 4):
            raise APIError(
                code="VALIDATION_ERROR",
                message=f"Channel at index {ch_idx} has invalid 'channel' number {ch_num}. Must be 1, 2, 3, or 4.",
                status_code=400
            )

        if ch_num in seen_channels:
            raise APIError(
                code="VALIDATION_ERROR",
                message=f"Duplicate channel number {ch_num} found in channels list.",
                status_code=400
            )
        seen_channels.add(ch_num)

        cleaned_ch = {"channel": ch_num}
        for metric in REQUIRED_CHANNEL_METRICS:
            if metric not in ch_data:
                raise APIError(
                    code="VALIDATION_ERROR",
                    message=f"Channel {ch_num} is missing required metric '{metric}'.",
                    status_code=400
                )
            val = ch_data[metric]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise APIError(
                    code="VALIDATION_ERROR",
                    message=f"Channel {ch_num} metric '{metric}' must be a finite number.",
                    status_code=400
                )
            cleaned_ch[metric] = float(val)

        cleaned_channels.append(cleaned_ch)

    # Sort channels by channel index 1..4
    cleaned_channels.sort(key=lambda x: x["channel"])

    # 5. Optional metadata with defaults
    sampling_rate_hz = payload.get("sampling_rate_hz", 20480.0)
    if isinstance(sampling_rate_hz, bool) or not isinstance(sampling_rate_hz, (int, float)) or not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise APIError("VALIDATION_ERROR", "Field 'sampling_rate_hz' must be a positive number.", status_code=400)

    sample_count = payload.get("sample_count", 20480)
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise APIError("VALIDATION_ERROR", "Field 'sample_count' must be a positive integer.", status_code=400)

    source = payload.get("source", "replayed_nasa_ims")
    if not isinstance(source, str) or not source.strip():
        raise APIError("VALIDATION_ERROR", "Field 'source' must be a non-empty string.", status_code=400)

    feature_count = payload.get("feature_count", 36)
    if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count != 36:
        raise APIError("VALIDATION_ERROR", "Field 'feature_count' must be 36.", status_code=400)

    raw_ingestion = payload.get("ingestion", {})
    if not isinstance(raw_ingestion, dict):
        raise APIError("VALIDATION_ERROR", "Field 'ingestion' must be a JSON object.", status_code=400)

    cleaned_ingestion = {
        "is_duplicate": bool(raw_ingestion.get("is_duplicate", False)),
        "is_out_of_order": bool(raw_ingestion.get("is_out_of_order", False)),
        "is_sequence_gap": bool(raw_ingestion.get("is_sequence_gap", False)),
        "processing_latency_ms": float(raw_ingestion.get("processing_latency_ms", 0.0))
        if isinstance(raw_ingestion.get("processing_latency_ms"), (int, float)) and math.isfinite(raw_ingestion.get("processing_latency_ms"))
        else 0.0,
        "is_replay_reset": bool(raw_ingestion.get("is_replay_reset", False))
    }

    dashboard_updated_at = payload.get("dashboard_updated_at")
    if dashboard_updated_at is not None:
        if isinstance(dashboard_updated_at, bool) or not isinstance(dashboard_updated_at, (int, float)) or not math.isfinite(dashboard_updated_at):
            dashboard_updated_at = time.time()
    else:
        dashboard_updated_at = time.time()

    return {
        "machine_id": clean_machine_id,
        "snapshot_sequence": seq,
        "original_timestamp": orig_ts.strip(),
        "ingest_timestamp": ingest_ts.strip(),
        "sampling_rate_hz": float(sampling_rate_hz),
        "sample_count": int(sample_count),
        "source": source.strip(),
        "feature_count": 36,
        "channels": cleaned_channels,
        "ingestion": cleaned_ingestion,
        "dashboard_updated_at": float(dashboard_updated_at)
    }


def save_live_telemetry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates, sequence-guards, and persists live telemetry state in MongoDB.
    
    Protections:
    - Rejects duplicate sequences and out-of-order snapshots (HTTP 409).
    - Preserves sequence ordering against concurrent race conditions via atomic conditional updates.
    - Permits verified replay-cycle resets (sequence 0 with is_replay_reset=True).
    - Fails with 503 on database unavailability or operational failures.
    """
    validated = validate_telemetry_payload(payload)
    machine_id = validated["machine_id"]
    new_seq = validated["snapshot_sequence"]
    ingestion = validated["ingestion"]
    is_replay_reset = ingestion["is_replay_reset"]

    # Ingestion metadata guards
    if ingestion["is_duplicate"]:
        raise APIError(
            code="DUPLICATE_SEQUENCE",
            message=f"Snapshot sequence {new_seq} is marked as duplicate by ingestion consumer.",
            status_code=409,
            details={"machine_id": machine_id, "sequence": new_seq}
        )

    if ingestion["is_out_of_order"]:
        raise APIError(
            code="OUT_OF_ORDER_SEQUENCE",
            message=f"Snapshot sequence {new_seq} is marked as out-of-order by ingestion consumer.",
            status_code=409,
            details={"machine_id": machine_id, "sequence": new_seq}
        )

    db = get_database()

    doc = {
        "_id": machine_id,
        "machine_id": machine_id,
        "snapshot_sequence": new_seq,
        "original_timestamp": validated["original_timestamp"],
        "ingest_timestamp": validated["ingest_timestamp"],
        "sampling_rate_hz": validated["sampling_rate_hz"],
        "sample_count": validated["sample_count"],
        "source": validated["source"],
        "feature_count": validated["feature_count"],
        "channels": validated["channels"],
        "ingestion": ingestion,
        "dashboard_updated_at": validated["dashboard_updated_at"],
        "updated_at": get_utc_now_iso()
    }

    try:
        existing = db.live_telemetry.find_one({"_id": machine_id})

        if existing is None:
            # First snapshot for this machine
            try:
                db.live_telemetry.insert_one(doc)
                result_doc = dict(doc)
                result_doc.pop("_id", None)
                return result_doc
            except DuplicateKeyError:
                # Concurrent request inserted first, re-fetch and proceed to update checks
                existing = db.live_telemetry.find_one({"_id": machine_id})

        if existing is not None:
            current_seq = int(existing.get("snapshot_sequence", -1))

            # Legitimate replay cycle reset: sequence 0 with verified is_replay_reset flag
            if new_seq == 0 and is_replay_reset:
                db.live_telemetry.update_one(
                    {"_id": machine_id},
                    {"$set": doc}
                )
            elif new_seq > current_seq:
                # Atomic conditional update requiring sequence to be strictly less than new_seq
                res = db.live_telemetry.update_one(
                    {"_id": machine_id, "snapshot_sequence": {"$lt": new_seq}},
                    {"$set": doc}
                )
                if res.matched_count == 0:
                    # Concurrently updated by another process
                    latest = db.live_telemetry.find_one({"_id": machine_id})
                    latest_seq = int(latest.get("snapshot_sequence", current_seq)) if latest else current_seq
                    if latest_seq == new_seq:
                        raise APIError(
                            code="DUPLICATE_SEQUENCE",
                            message=f"Snapshot sequence {new_seq} was concurrently recorded for machine '{machine_id}'.",
                            status_code=409,
                            details={"machine_id": machine_id, "sequence": new_seq, "current_sequence": latest_seq}
                        )
                    else:
                        raise APIError(
                            code="OUT_OF_ORDER_SEQUENCE",
                            message=f"Snapshot sequence {new_seq} was superseded by concurrent sequence {latest_seq}.",
                            status_code=409,
                            details={"machine_id": machine_id, "sequence": new_seq, "current_sequence": latest_seq}
                        )
            elif new_seq == current_seq:
                raise APIError(
                    code="DUPLICATE_SEQUENCE",
                    message=f"Snapshot sequence {new_seq} has already been recorded for machine '{machine_id}'.",
                    status_code=409,
                    details={"machine_id": machine_id, "sequence": new_seq, "current_sequence": current_seq}
                )
            else:
                raise APIError(
                    code="OUT_OF_ORDER_SEQUENCE",
                    message=f"Snapshot sequence {new_seq} is older than current recorded sequence {current_seq}.",
                    status_code=409,
                    details={"machine_id": machine_id, "sequence": new_seq, "current_sequence": current_seq}
                )

        result_doc = dict(doc)
        result_doc.pop("_id", None)
        return result_doc

    except APIError:
        raise
    except PyMongoError as err:
        logger.error(f"MongoDB operational error while persisting telemetry: {err}", exc_info=True)
        raise APIError("DATABASE_ERROR", "Database error occurred while persisting telemetry.", status_code=503)


def get_latest_telemetry(machine_id: str) -> Dict[str, Any]:
    """
    Retrieves the latest persisted telemetry state for a machine.
    Read-only query returning the stored state dictionary.
    """
    clean_id = validate_machine_id(machine_id)
    db = get_database()

    try:
        doc = db.live_telemetry.find_one({"_id": clean_id})
    except PyMongoError as err:
        logger.error(f"MongoDB operational error while reading telemetry: {err}", exc_info=True)
        raise APIError("DATABASE_ERROR", "Database error occurred while reading telemetry.", status_code=503)

    if not doc:
        raise APIError(
            code="TELEMETRY_NOT_FOUND",
            message=f"No telemetry recorded yet for machine '{clean_id}'.",
            status_code=404,
            details={"machine_id": clean_id}
        )

    doc_dict = dict(doc)
    doc_dict.pop("_id", None)
    return doc_dict
