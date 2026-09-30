"""
ml_service.py - ML Integration layer for MachineMind AI.

ONLY module in the backend that imports and interacts with AnomalyInferenceEngine.
Maintains isolated per-machine engine instances to prevent persistence state corruption across machines.
"""

import sys
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from backend.utils.responses import get_utc_now_iso
from backend.utils.errors import APIError

logger = logging.getLogger(__name__)

# State variables for ML service singleton
_ml_service_initialized = False
_model_dir: Optional[Path] = None
_model_version: str = "v1"
_machine_engines: Dict[str, Any] = {}


def setup_ml_path(ml_service_path: str) -> None:
    """Adds ML service path to sys.path if not already present."""
    resolved_path = str(Path(ml_service_path).resolve())
    if resolved_path not in sys.path:
        sys.path.insert(0, resolved_path)


def init_ml_service(model_dir: str, ml_service_path: str, model_type: str = "iforest") -> bool:
    """
    Initializes the ML service singleton and verifies model artifacts can be loaded.
    Returns True if initialized successfully, False otherwise.
    """
    global _ml_service_initialized, _model_dir, _model_version, _machine_engines
    try:
        setup_ml_path(ml_service_path)
        from src.inference import AnomalyInferenceEngine

        _model_dir = Path(model_dir).resolve()
        if not _model_dir.exists():
            logger.error(f"Model directory not found: {_model_dir}")
            _ml_service_initialized = False
            return False

        # Attempt to load a temporary verification engine instance to confirm artifacts are valid
        test_engine = AnomalyInferenceEngine(artifacts_dir=_model_dir, model_type=model_type, stateful=True)
        _model_version = test_engine.metadata.get("pipeline_version", "v1")
        _ml_service_initialized = True
        logger.info(f"ML Service successfully initialized with model_version='{_model_version}' from {_model_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize ML Service: {e}", exc_info=True)
        _ml_service_initialized = False
        return False


def is_model_loaded() -> bool:
    """Returns True if ML models are loaded and available for inference."""
    return _ml_service_initialized


def get_model_version() -> str:
    """Returns current ML model version string."""
    return _model_version


def get_engine_for_machine(machine_id: str, model_type: str = "iforest") -> Any:
    """
    Retrieves or creates an isolated AnomalyInferenceEngine instance for the given machine_id.
    Ensures per-machine persistence counters remain isolated.
    """
    if not _ml_service_initialized or _model_dir is None:
        raise APIError("MODEL_UNAVAILABLE", "ML inference engine is not loaded or unavailable.", status_code=503)

    if machine_id not in _machine_engines:
        from src.inference import AnomalyInferenceEngine
        logger.info(f"Creating new AnomalyInferenceEngine instance for machine_id='{machine_id}'")
        engine = AnomalyInferenceEngine(
            artifacts_dir=_model_dir,
            model_type=model_type,
            stateful=True
        )

        # Attempt rehydrating consecutive exceedance counters from latest Mongo prediction
        try:
            from flask import current_app
            from backend.utils.db import get_db
            if current_app:
                db = get_db(current_app.config["MONGO_URI"], current_app.config["DATABASE_NAME"])
                if db is not None:
                    latest_pred = db.predictions.find_one({"machine_id": machine_id}, sort=[("timestamp", -1)])
                    if latest_pred and "channels" in latest_pred:
                        for ch_info in latest_pred["channels"]:
                            ch_num = ch_info.get("channel", 1)
                            ch_key = f"ch{ch_num}"
                            if ch_key in engine.consecutive_exceedances:
                                engine.consecutive_exceedances[ch_key] = int(ch_info.get("consecutive_flagged_count", 0))
                        logger.info(f"Rehydrated engine state for machine_id='{machine_id}': {engine.consecutive_exceedances}")
        except Exception as e:
            logger.debug(f"State rehydration skipped for machine_id='{machine_id}': {e}")

        _machine_engines[machine_id] = engine

    return _machine_engines[machine_id]


def run_inference(
    machine_id: str,
    snapshot: np.ndarray,
    timestamp: Optional[str] = None,
    source_filename: Optional[str] = None,
    model_type: str = "iforest"
) -> Dict[str, Any]:
    """
    Runs end-to-end ML inference on a validated 20480x4 raw snapshot for machine_id.
    Maps AnomalyInferenceEngine output to internal API response dict.
    """
    engine = get_engine_for_machine(machine_id, model_type=model_type)

    try:
        raw_result = engine.predict_snapshot(snapshot, timestamp=timestamp)
    except Exception as e:
        logger.error(f"Inference execution failed for machine_id='{machine_id}': {e}", exc_info=True)
        raise APIError("INTERNAL_ERROR", f"ML inference failed: {str(e)}", status_code=500)

    # Channel mappings
    ch_keys = ["ch1", "ch2", "ch3", "ch4"]
    channels_output = []

    for idx, ch_key in enumerate(ch_keys, start=1):
        score_val = raw_result["channel_scores"][ch_key]
        thresh_val = raw_result["channel_thresholds"][ch_key]
        snap_flag = raw_result["channel_raw_flags"][ch_key]
        count_val = raw_result["state_consecutive_counts"][ch_key]
        sust_flag = raw_result["channel_sustained_flags"][ch_key]

        if sust_flag:
            ch_state = "anomaly_detected"
        elif snap_flag:
            ch_state = "watch"
        else:
            ch_state = "normal"

        channels_output.append({
            "channel": idx,
            "name": f"Channel {idx}",
            "anomaly_score": float(score_val),
            "threshold": float(thresh_val),
            "snapshot_flagged": bool(snap_flag),
            "consecutive_flagged_count": int(count_val),
            "persistence_confirmed": bool(sust_flag),
            "state": ch_state
        })

    overall_snap_flagged = any(raw_result["channel_raw_flags"].values())
    overall_pers_confirmed = bool(raw_result["system_alert"])

    if overall_pers_confirmed:
        overall_state = "anomaly_detected"
        overall_label = "Vibration anomaly detected"
    elif overall_snap_flagged:
        overall_state = "watch"
        overall_label = "Anomalous snapshot — awaiting persistence confirmation"
    else:
        overall_state = "normal"
        overall_label = "Normal"

    prediction_record = {
        "prediction_id": str(uuid.uuid4()),
        "machine_id": machine_id,
        "timestamp": timestamp or get_utc_now_iso(),
        "model_version": _model_version,
        "source_filename": source_filename,
        "overall": {
            "state": overall_state,
            "snapshot_flagged": overall_snap_flagged,
            "persistence_confirmed": overall_pers_confirmed,
            "label": overall_label
        },
        "channels": channels_output,
        "persistence": {
            "required_consecutive_snapshots": getattr(engine, "k", 3)
        },
        "alert": {
            "created": False,
            "alert_id": None,
            "severity": None
        }
    }

    return prediction_record
