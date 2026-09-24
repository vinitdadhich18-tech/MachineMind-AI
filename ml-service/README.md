# MachineMind AI - ML Service

Predictive Maintenance System for Rotating Machinery - Machine Learning Component.

## Directory Structure

- `data/raw/`: Raw, unprocessed sensor logs and vibration metrics.
- `data/processed/`: Cleaned, feature-engineered datasets ready for modeling.
- `notebooks/`: Jupyter Notebooks for exploratory data analysis (EDA) and prototyping.
- `models/`: Trained ML model artifacts and serializations (e.g., `.pkl`, `.joblib`).
- `reports/`: Generated analysis reports, performance metrics, and plots.
- `src/`: Core Python modules and source scripts.
- `requirements.txt`: Python package dependencies.
- `src/check_environment.py`: Script to verify installed packages and versions.

## Setup Instructions

1. Navigate to the `ml-service` directory:
   ```bash
   cd ml-service
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Verify the environment setup:
   ```bash
   python src/check_environment.py
   ```
