# Rock vs Mine Prediction Using Logistic Regression

## Problem Statement
The objective of this project is to build a machine learning model that classifies sonar signals as either a rock or a mine based on signal frequency data.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Logistic Regression

## Approach
- Loaded the sonar dataset containing 60 numerical features
- Separated features and target labels
- Split the data into training and testing sets
- Trained a Logistic Regression classifier
- Evaluated model performance using accuracy metric

## Project Structure
- `dataset/` – Contains the sonar dataset
- `src/`
  - `data_preprocessing.py` – Data loading and train-test split
  - `model.py` – Logistic Regression model definition
  - `train.py` – Model training and evaluation
- `requirements.txt` – Project dependencies
- `README.md` – Project documentation

## Results
- Achieved an accuracy of **76.1%** on the test dataset

## How to Run
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the training script:
   ```bash
   python src/train.py
