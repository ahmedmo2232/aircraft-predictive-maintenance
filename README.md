# Aircraft Predictive Maintenance: Remaining Useful Life Prediction

Predicting how many flight cycles a turbofan engine has left before failure, using NASA's C-MAPSS engine degradation dataset (FD001).



---

## The problem

Jet engines don't fail without warning. Long before a breakdown, sensor readings start to drift: temperatures creep up, pressures shift, efficiency falls. **Predictive maintenance** uses that drift to estimate how much life a machine has left, so it can be serviced before it fails instead of after.

This project answers two questions about an engine, using only its sensor history:

1. **How many flight cycles does it have left?** A regression problem: predict the Remaining Useful Life (RUL) as a number.
2. **Does it need maintenance soon?** A classification problem: will it fail within the next 30 cycles, yes or no? This is the decision a maintenance planner actually acts on.

## Dataset

[NASA C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) (Commercial Modular Aero-Propulsion System Simulation), subset **FD001**:

- 100 training engines, each run from healthy until failure (20,631 cycles in total)
- 100 test engines, each cut off at a random point before failure
- 21 sensor readings and 3 operating settings per cycle
- One operating condition, one fault mode (high-pressure compressor degradation)

## Approach

**1. Data quality.** Detected and removed sensors with zero variance, which carry no information.

**2. Target engineering.** Built the RUL target from each engine's failure cycle and **capped it at 125 cycles**. A new engine and a slightly used one look identical to the sensors, so asking a model to tell "300 cycles left" from "250 cycles left" means asking it to learn noise. The cap is a standard choice for this dataset.

**3. Exploratory analysis.** Fleet-averaged sensor trends showed clear, consistent drift as failure approaches. Individual readings were noisy, which motivated the feature engineering below.

**4. Feature engineering.** For every sensor, computed per engine:
- **Rolling means** (5 and 20 cycles), which smooth out noise
- **Rolling standard deviations** (5 and 20 cycles), since erratic readings are a symptom of wear
- **Drift from the engine's first reading**, which removes manufacturing differences between engines and isolates wear

All features use only past and present data for each engine, so there is no leakage from the future.

**5. Modelling.**
- *Regression:* Baseline (mean), Linear Regression, Random Forest, XGBoost
- *Classification* (RUL ≤ 30 cycles): Logistic Regression, Random Forest, with `class_weight="balanced"` to handle the 85/15 class imbalance

## Results

### Regression: predicting remaining life

| Model | MAE (cycles) | RMSE (cycles) | R² |
|---|---|---|---|
| Baseline (mean) | 34.83 | 41.94 | −0.10 |
| Linear Regression | 13.18 | 16.51 | 0.83 |
| **Random Forest** | 9.46 | **12.46** | **0.90** |
| **XGBoost** | **9.13** | 12.56 | **0.90** |

- The best models estimate remaining life to within about **9 cycles on average** (RMSE 12.5).
- That is a **70% lower RMSE** than predicting the fleet average.
- Random Forest and XGBoost are effectively tied: the difference is a fraction of a cycle on 100 test engines.

### Classification: will the engine fail within 30 cycles?

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.990 | 0.962 | 1.000  | 0.980 |
| Random Forest | 0.950 | 0.917 | 0.880 | 0.898 |

The final classifier was selected on **recall**: in aviation, a missed failure is far more costly than a false alarm. The best model which was the logistic regression caught 25 out of 25 engines that genuinely needed maintenance.


## Key findings

- **The sensors carry a strong predictive signal.** Every model beat the baseline by a wide margin.
- **Feature engineering mattered most.** Rolling and drift features dominated the feature-importance rankings over the raw sensor readings.
- **Regression and classification answer different questions.** Classification exposes the asymmetry regression metrics hide: a false alarm costs an inspection, a missed failure costs far more.
-**Which model works best for each problem** with regression based trees models worked better, however for the classification the logistic regression worked better. 

## Limitations

- FD001 is the simplest subset (one operating condition, one fault mode). Results would be lower on FD002–FD004.
- The data is simulated, so it is cleaner than real engine telemetry.
- Test RUL values were capped at 125 to match the training target. This is standard practice for comparability, but it makes scores somewhat more favourable.
- Each test engine is evaluated at a single cut-off point.
- Hyperparameters were set by hand, not tuned.

## How to run

1. Open `NASA_CMAPSS_RUL_Predictive_Maintenance.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Add your Kaggle credentials (Kaggle → Settings → API → Create New Token) as Colab Secrets named `KAGGLE_USERNAME` and `KAGGLE_KEY`, or upload `kaggle.json` when prompted.
3. Run all cells from top to bottom. The dataset downloads automatically.

**Tools:** Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost

## Notes

- This project was done for a data analytics class, the code is heavily commented which was required by the class professor. 



**Dataset citation:** A. Saxena and K. Goebel (2008). *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Prognostics Data Repository.

**Author:** Ahmed Abotrika, BSc Artificial Intelligence, Lusail University
