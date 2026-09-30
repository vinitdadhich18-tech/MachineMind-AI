# MachineMind AI — Flask Backend

Flask REST API application layer for MachineMind AI unsupervised vibration anomaly detection.

## Setup & Running

### Environment
Ensure Python 3.14.2 is installed. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### Installation
```bash
pip install -r requirements.txt
```

### Running Server
```bash
python app.py
```

### Running Tests
```bash
python -m pytest backend/tests
```
