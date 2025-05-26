# End-to-End Machine Learning Project: Heart Disease Prediction

## Project Objective

This project demonstrates a complete machine learning workflow, from data preprocessing and training to model evaluation and deployment. The goal is to build a reliable model to predict heart disease based on patient data, while integrating best practices such as logging, experiment tracking with MLflow, and visualization through a Streamlit web app.

---
## Project Structure

```plaintext
e2e-ML-project/
├── app/
│ └── app.py
├── src/
│ ├── train.py 
│ ├── evaluate.py 
│ ├── mlflow_tracking.py 
│ └── data_preprocessing.py 
├── artifacts/
│ ├── models/ 
│ └── images/ 
├── data/
│ └── heart.csv 
├── logs/
│ └── pipeline.log 
├── dockerfiles/
│ ├── Dockerfile_train 
│ ├── Dockerfile_evaluate 
│ ├── Dockerfile_mlflow 
│ └── Dockerfile_streamlit 
├── config.py 
├── requirements.txt 
└── docker-compose.yaml 
```
---
## Dataset Description

The dataset used (`data/heart.csv`) contains clinical features collected from patients and is used to predict the presence of heart disease. Below are the descriptions of each column:

| Column         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| Age            | Age of the patient (years)                                                  |
| Sex            | Sex of the patient (M or F)                                                 |
| ChestPainType  | Type of chest pain (Typical, Atypical, Non-anginal, Asymptomatic)          |
| RestingBP      | Resting blood pressure (in mm Hg)                                           |
| Cholesterol    | Serum cholesterol level (in mg/dl)                                          |
| FastingBS      | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)                       |
| RestingECG     | Resting electrocardiogram results (Normal, ST, LVH)                         |
| MaxHR          | Maximum heart rate achieved during exercise                                 |
| ExerciseAngina | Exercise-induced angina (Y = yes; N = no)                                   |
| Oldpeak        | ST depression induced by exercise relative to rest                          |
| ST_Slope       | Slope of the peak exercise ST segment (Up, Flat, Down)                      |
| HeartDisease   | Target variable (1 = presence of heart disease; 0 = absence of disease)     |
                       |

---

## Technologies Used

- **Python 3.12** – Programming language
- **Pandas, NumPy, Scikit-learn** – Data preprocessing and ML modeling
- **Matplotlib, Seaborn** – Data visualization
- **MLflow** – Experiment tracking and model registry
- **Streamlit** – Web-based visualization app
- **Docker** – Containerization of each component
- **Docker Compose** – Orchestration of all services
- **Logging** – To trace pipeline execution

---

---

## How to Run the Project

### 1. Clone the Repository

```
git clone https://github.com/your-username/e2e-ML-project.git
cd e2e-ML-project
```
### 2. Build Docker Images
```
docker-compose build
```
### 3. Run All Services
```
docker-compose up
```
This will start:
- **train_service**: Trains the model and logs metrics to MLflow.
- **evaluate_service**: Evaluates the model and generate plots
- **mlflow_service**: Starts MLflow UI at http://localhost:5000
- **streamlit_service**: Starts Streamlit app at http://localhost:8501
### 4. To stop all services
```
docker-compose down
```
### 5. Individual Services Execution
If you want to run individual services, you can do so by executing the following commands:
- **Train Service**: 
```
docker-compose run train
```
- **Evaluate Service**: 
```
docker-compose run evaluate
```
- **MLflow Service**: 
```
docker-compose run mlflow
```
- **Streamlit Service**: 
```
docker-compose run streamlit
```
### Notes
- Make sure Docker and Docker Compose are installed on your machine.
- All artifacats will be saved in the `artifacts/` directory.




