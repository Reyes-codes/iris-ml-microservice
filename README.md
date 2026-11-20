# Iris ML Microservice

A machine learning microservice that classifies Iris flowers into three species using four numerical measurements. The project combines a Scikit-learn Decision Tree model, a REST API built with FastAPI, and a web interface for interactive predictions.

## Overview

This project implements a classification system capable of identifying three Iris flower species:

- Setosa
- Versicolor
- Virginica

The model uses four input measurements, expressed in centimeters:

| Feature | Description |
|---|---|
| `sepal_length` | Sepal length |
| `sepal_width` | Sepal width |
| `petal_length` | Petal length |
| `petal_width` | Petal width |

The application exposes the trained model through a REST API and provides a web interface where users can enter measurements and obtain predictions.

## Model

The classifier was developed using the Iris dataset included in Scikit-learn.

**Algorithm:** Decision Tree Classifier

**Configuration:**
- `max_depth=3`
- `random_state=42`
- Training set: 80%
- Testing set: 20%
- Stratified train-test split

**Test accuracy:** 96.67%

The trained model is serialized using Joblib and stored as `iris_model.pkl`, together with the target class names.

## Technologies

- Python
- Scikit-learn
- FastAPI
- Uvicorn
- Pydantic
- Joblib
- HTML
- CSS
- JavaScript
- Git and GitHub

## Project Structure

```text
iris-ml-microservice/
│
├── app.py
├── train_model.py
├── iris_model.pkl
├── requirements.txt
├── .gitignore
│
└── static/
    └── index.html
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Reyes-codes/iris-ml-microservice.git
cd iris-ml-microservice
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train the model (optional)

The repository includes the trained model artifact. To train it again, execute:

```bash
python train_model.py
```

This generates the `iris_model.pkl` file.

### 5. Start the API

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open the application in your browser:

http://127.0.0.1:8000

## API Documentation

FastAPI automatically generates interactive API documentation.

Once the server is running, access:

http://127.0.0.1:8000/docs

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the web interface |
| GET | `/health` | Checks API availability |
| POST | `/predict` | Predicts the Iris flower species |

### Prediction Example

**Request:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response structure:**

The API returns the predicted species, its class ID, confidence, class probabilities, and inference time.

```json
{
  "prediction": "setosa",
  "class_id": 0,
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "inference_ms": 0.5
}
```

*The response above illustrates the expected structure; actual values depend on the input.*

## Web Interface

The project includes a web interface that allows users to:

- Enter the four Iris measurements.
- Select example inputs.
- Submit measurements to the prediction API.
- View the predicted species and class probabilities.
- Check prediction timing information.

## Learning Objectives

This project demonstrates practical skills in:

- Supervised machine learning and classification.
- Model training, evaluation, and serialization.
- REST API development with FastAPI.
- Integrating a machine learning model into a web application.
- Input validation with Pydantic.
- Version control and project management with Git and GitHub.

## Author

**Reyes-codes**

GitHub: https://github.com/Reyes-codes

---

*Academic project focused on machine learning model deployment and REST API development.*
