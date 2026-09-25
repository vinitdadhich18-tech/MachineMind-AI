# Feature Engineering Notes & Report — NASA IMS Bearing Dataset (Set 2)

**Project:** MachineMind AI  
**Component:** `ml-service`  
**Phase:** Phase 5 — Feature Engineering  
**Date:** 2026-09-26  
**Document Status:** Complete Feature Extraction & Strategy Specification  

---

## 1. Executive Summary

This report documents the design, implementation, mathematical conventions, and validation results of the **Feature Engineering Pipeline** for **Set 2** of the **NASA IMS Bearing Dataset**.

The objective of Phase 5 is to transform high-dimensional raw vibration waveforms ($81,920$ sample points per snapshot across 4 channels) into a compact, low-dimensional, physically interpretable statistical feature matrix ($984 \text{ rows} \times 40 \text{ columns}$) without applying unapproved digital filtering, per-snapshot normalization, or predictive model training.

---

## 2. Feature Extraction Definitions & Conventions

### 2.1 Time-Domain Features (7 Features per Channel)
For a discrete vibration sample vector $x = [x_1, x_2, \dots, x_N]$ of length $N$:

1. **Mean ($\mu$):**
   $$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$
   *Definition:* Arithmetic mean of signal amplitude, measuring central DC baseline offset.
2. **Sample Standard Deviation ($\sigma$):**
   $$\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu)^2}$$
   *Definition:* Sample standard deviation ($N-1$ degrees of freedom, `ddof=1`), measuring AC signal dispersion around the mean.
3. **Root Mean Square (RMS):**
   $$x_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$
   *Definition:* Total dynamic energy contained within the signal ($\text{RMS}^2 = \mu^2 + \sigma_{\text{pop}}^2$).
4. **Peak-to-Peak Amplitude ($P2P$):**
   $$P2P = \max(x) - \min(x)$$
   *Definition:* Absolute dynamic range between maximum positive peak and minimum negative trough.
5. **Skewness ($S$):**
   $$S = \frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^3}{\sigma^3}$$
   *Definition:* Third standardized moment, measuring distribution asymmetry around the mean ($S = 0.0$ for symmetric Gaussian noise).
6. **Fisher Excess Kurtosis ($K$):**
   $$K = \frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^4}{\sigma^4} - 3.0$$
   *Definition:* Fourth standardized moment minus $3.0$ (Fisher's excess kurtosis). Pure Gaussian noise evaluates to $K = 0.0$. Positive values ($K > 0.0$) indicate impulsive shock spikes caused by micro-spalling.
7. **Crest Factor ($CF$):**
   $$CF = \frac{\max(|x|)}{x_{\text{rms}} + \epsilon}$$
   *Definition:* Ratio of maximum peak magnitude to RMS energy, guarded against zero-RMS signals by $\epsilon = 10^{-12}$. Measures signal impulsiveness relative to background energy.

---

### 2.2 Frequency-Domain Features (2 Features per Channel)
Using real-valued Fast Fourier Transform (FFT) with a **Hanning window** $w(n) = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$:

1. **Spectral Energy ($E_{\text{spectral}}$):**
   $$E_{\text{spectral}} = \sum_{k=1}^{N/2} |X(f_k)|^2$$
   *Definition:* Sum of squared normalized positive-frequency FFT magnitudes (excluding the 0 Hz DC bin).  
   > **Methodological Note:** Represents a windowed spectral feature, NOT calibrated physical power ($W$).
2. **Spectral Centroid ($f_{\text{centroid}}$):**
   $$f_{\text{centroid}} = \frac{\sum_{k=1}^{N/2} f_k \cdot |X(f_k)|}{\sum_{k=1}^{N/2} |X(f_k)| + \epsilon}$$
   *Definition:* Center-of-mass frequency ($Hz$) of the positive FFT magnitude spectrum.

---

## 3. Dataset Schema & Matrix Dimensions

The exported feature dataset is stored at [`data/processed/features_set2.csv`](file:///d:/Projects/MachineMind%20AI/ml-service/data/processed/features_set2.csv).

- **Matrix Dimensions:** **984 rows $\times$ 40 columns** (4 metadata columns + 36 feature columns).
- **Metadata Columns (4):** `file_index`, `filename`, `timestamp`, `is_valid`.
- **Feature Columns (36):** For each channel $c \in [1, 2, 3, 4]$:
  `ch{c}_mean`, `ch{c}_std`, `ch{c}_rms`, `ch{c}_p2p`, `ch{c}_skewness`, `ch{c}_kurtosis`, `ch{c}_crest_factor`, `ch{c}_spectral_energy`, `ch{c}_spectral_centroid`.

---

## 4. Empirical Validation Results

Synthetic unit tests in [`src/test_feature_extraction.py`](file:///d:/Projects/MachineMind%20AI/ml-service/src/test_feature_extraction.py) passed 100% of test cases:
1. **Zero Signal ($x=0$):** Mean = 0, Std = 0, RMS = 0, Crest Factor = 0.0, zero NaNs/Infs.
2. **Constant Signal ($x=5.0$):** Mean = 5.0, Std = 0, P2P = 0, Crest Factor = 1.0, zero NaNs/Infs.
3. **Pure Sine Wave ($1000 \text{ Hz}$):** RMS $\approx 0.7071$, Crest Factor $\approx 1.4142$, Spectral Centroid $\approx 1000.0 \text{ Hz}$.
4. **Gaussian Noise ($\mathcal{N}(0, 1)$):** Fisher Kurtosis $\approx 0.0$, Skewness $\approx 0.0$.
5. **Impulsive Spike Signal:** Fisher Kurtosis $> 5.0$, Crest Factor $> 3.0$.
6. **Exception Handling:** Invalid inputs (non-DataFrame, NaN, Inf, negative $f_s$) raise `TypeError` / `ValueError`.
7. **Schema & Finiteness:** Verified exact 36 feature key generation and $100\%$ finite float values.

---

## 5. Methodological Safeguards & Limitations

1. **Unverified Sampling Rate:** Sampling rate $f_s = 20,480 \text{ Hz}$ is based on dataset documentation; not independently verified from binary metadata.
2. **Uncalibrated Sensor Units:** Absolute feature values are uncalibrated voltages/acceleration readings. All downstream anomaly detection models must rely on relative statistical growth trends from the healthy baseline period.
3. **Neutral Handling of End-of-Test Low-Amplitude Snapshots:** The final two low-amplitude snapshots are processed neutrally without asserting machine shutdown or failure timing.
4. **No Predictive Claims:** Features describe statistical waveform properties; no feature is claimed to independently predict bearing failure or remaining useful life (RUL).
