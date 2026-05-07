# Credit Card Fraud Detection

End-to-end machine learning pipeline for detecting fraudulent credit card transactions, from exploratory analysis to model training and REST API deployment.

## Dataset

The dataset contains **284,807** credit card transactions made by European cardholders in September 2013 (available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)).

| Metric | Value |
|--------|-------|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.17%) |
| Features | 30 (Time, Amount, V1–V28 PCA-transformed) |
| Target | `Class` : 0 = legitimate, 1 = fraud |

## Project Structure

```
credit_card_fraud_detection/
├── api/
│   ├── app.py              # Flask REST API
│   └── logger.py           # Structured logging with daily rotation
├── data/
│   ├── raw/                # Original dataset (not tracked)
│   └── processed/          # Cleaned dataset (not tracked)
├── logs/                   # Runtime API logs (daily rotation)
├── models/
│   ├── best_model_*.joblib # Serialized best model
│   └── model_metadata.json # Threshold, features, metrics
├── notebooks/
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   └── train_and_save.ipynb# Model training, evaluation & export
├── results/
│   └── model_comparison_results.csv
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Models Evaluated

### Supervised Models
Trained with three sampling strategies (baseline, SMOTE, undersampling):

| Model | Strategy | AUPRC | F1 |
|-------|----------|-------|----|
| **Random Forest** | **SMOTE** | **0.862** | **0.859** |
| Random Forest | Baseline | 0.861 | 0.847 |
| Random Forest | Under | 0.845 | 0.795 |
| Logistic Regression | SMOTE | 0.765 | 0.781 |

### Unsupervised Model
- **Isolation Forest**  trained only on legitimate transactions, detects novel anomalies without labels

The best model (Random Forest + SMOTE) is saved and served via the API.

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone <repo-url>
cd credit_card_fraud_detection

conda create -n fraud-detection python=3.11
conda activate fraud-detection
pip install -r requirements.txt
```

### Run the API Locally

```bash
python api/app.py
```

### Run with Docker

```bash
docker build -t fraud-detection-api .
docker run -p 5001:5001 fraud-detection-api
```

### Test the API

**Health check:**
```bash
curl http://localhost:5001/health
```

**Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0, "Amount": 100,
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
    "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
    "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
    "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
    "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
    "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
    "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
    "V28": -0.021053
  }'
```

**Response:**
```json
{
  "fraud_probability": 0.005,
  "fraud_prediction": 0,
  "threshold_used": 0.46
}
```

## Streamlit Dashboard

An interactive dashboard for exploring model performance, making predictions, and analyzing data.

```bash
streamlit run streamlit_app.py
```

**Pages:**
- **Dashboard**: Model metrics, AUPRC/F1 comparison charts, precision-recall scatter
- **Predict**: Real-time fraud prediction with gauge visualization (manual form or JSON input)
- **Data Explorer**: Dataset statistics, class distribution, amount distributions, feature correlations

## Logging

The API logs every request with structured information:

```
2026-03-16 16:40:16 | INFO | REQUEST | ip=192.168.1.10 | method=POST | path=/predict
2026-03-16 16:40:16 | INFO | PREDICTION | ip=192.168.1.10 | proba=0.005000 | prediction=0
2026-03-16 16:40:16 | INFO | RESPONSE | ip=192.168.1.10 | method=POST | path=/predict | status=200
```

- Logs rotate daily with 90-day retention
- Log files: `logs/api_YYYY-MM-DD.log`
- Captures: client IP, request method, path, predictions, errors with tracebacks

## Tech Stack

- **ML**: scikit-learn, imbalanced-learn, pandas, numpy
- **API**: Flask
- **Dashboard**: Streamlit, Plotly
- **Deployment**: Docker
- **Logging**: Python logging with TimedRotatingFileHandler
