# MachineMind AI — Phase 3: InfluxDB v2 Integration & Time-Series Schema Documentation

**Module**: `ml-service/src/influx_writer.py`  
**Test Suite**: `ml-service/src/test_phase3_influx.py`  
**Infrastructure**: `docker-compose.yml` (InfluxDB v2.7 on port `8086`)  
**Bucket**: `machinemind_telemetry` | **Org**: `machinemind_org` | **Default Retention**: `30d`  
**Date**: October 1, 2026  

---

## 1. Executive Summary & Architectural Role

In Phase 3, **InfluxDB v2** is integrated as the primary time-series database for MachineMind AI.

```
NASA IMS Dataset
       │
       ▼
TelemetrySimulator (Replay Engine)
       │
       ▼ (MQTT over TLS, QoS 1)
HiveMQ Cloud Broker
       │
       ▼
TelemetryIngestionConsumer (Phase 2 Boundary)
       │
       ▼
  TelemetryRecord
       │
       ▼
InfluxDBWriter ──> InfluxDB v2 ('machinemind_telemetry')
       │
       ▼ (Future Phase 7)
    Grafana Dashboards
```

---

## 2. InfluxDB v2 Line Protocol Schema Specification

### Measurement 1: `vibration_features`
Stores 7 time-domain and 2 frequency-domain summary features per channel per snapshot.

- **Tags (Low-Cardinality Indexed Metadata)**:
  - `machine_id`: Monitored machinery identifier (e.g. `ims_set2_rig`).
  - `channel`: Channel name (`ch1`, `ch2`, `ch3`, `ch4`).
  - `source`: Telemetry provenance (`replayed_nasa_ims`).

- **Fields (Numeric Feature Measurements & Metadata)**:
  - `snapshot_sequence`: Monotonic integer sequence ID.
  - `original_timestamp`: Original NASA snapshot timestamp string (`YYYY-MM-DDTHH:MM:SSZ`).
  - `mean`: Arithmetic baseline mean offset.
  - `std`: Sample standard deviation ($ddof=1$).
  - `rms`: Root Mean Square amplitude $\sqrt{\text{mean}(x^2)}$ (total signal energy).
  - `p2p`: Peak-to-peak amplitude $\max(x) - \min(x)$.
  - `skewness`: Third standardized moment.
  - `kurtosis`: Fisher excess kurtosis ($\text{Gaussian} = 0.0$).
  - `crest_factor`: Peak-to-RMS ratio.
  - `spectral_energy`: Hann-windowed positive frequency FFT energy sum.
  - `spectral_centroid`: Center of mass frequency (Hz).

- **Point Timestamp**:
  - `ingest_timestamp` (mapped to virtual live timeline relative to `now()`) is used as the InfluxDB point time.
  - Allows Grafana relative time ranges ("Last 24 hours") to work smoothly during live replays.

---

### Measurement 2: `pipeline_health`
Stores ingestion pipeline operational metrics for Grafana observability.

- **Tags**:
  - `service_name`: `mqtt_ingestion_consumer`
  - `machine_id`: Monitored equipment ID

- **Fields**:
  - `snapshot_sequence`: Monotonic integer sequence ID.
  - `processing_latency_ms`: Ingestion consumer processing duration (ms).
  - `is_duplicate`: Integer flag ($0$ or $1$).
  - `is_out_of_order`: Integer flag ($0$ or $1$).
  - `is_sequence_gap`: Integer flag ($0$ or $1$).

---

## 3. Raw Waveform Archival Decision

- **Decision**: Raw $20,480 \times 4$ floating-point sample arrays ($\sim 510\text{ KB}$ per snapshot) are **NOT** stored as individual fields in InfluxDB.
- **Rationale**: Storing 81,920 individual floats per snapshot inside a time-series database creates extreme write amplification and memory bloat. InfluxDB stores engineered vibration features ($\text{RMS}$, $\text{P2P}$, Spectral Centroid, etc.) while the raw vibration files remain archived in the raw dataset directory.

---

## 4. Idempotency & Duplicate Handling

- If `record.ingestion_metadata['is_duplicate'] == True` (triggered by MQTT QoS 1 retry delivery), `InfluxDBWriter.write_telemetry_record()` skips writing points to InfluxDB.
- Prevents duplicate time-series points from inflating metrics or skewing feature trends.

---

## 5. Viva Defense Notes (Plain Language Explanation)

> **Why did we choose InfluxDB for MachineMind AI time-series storage?**  
> "Relational databases (PostgreSQL) and document stores (MongoDB) are not optimized for high-frequency sequential telemetry queries. InfluxDB is built specifically for time-series data: it compresses numeric metrics efficiently, indexes tags (`machine_id`, `channel`) for instant Flux query responses, and integrates directly with Grafana without requiring custom backend API endpoints."

---

## 6. Verification Evidence

- **Unit Tests**: `4/4 passed in 2.10s` (`ml-service/src/test_phase3_influx.py`)
- **Workspace Test Suite**: `45/45 passed in 7.82s` (backend, frontend, phase 1, phase 2, phase 3)
- **Live Integration Test Result**: 3 real NASA IMS snapshot payloads pub/subbed over HiveMQ Cloud TLS and written to InfluxDB schema (5 points per batch: 4 channel feature points + 1 pipeline health point).
