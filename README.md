# Diabetes Risk Prediction — Machine Learning Project

## Project Overview
This project builds a supervised machine learning model to predict whether a patient is likely to have diabetes based on diagnostic health measurements.

The goal is to demonstrate an end-to-end machine learning workflow: data loading, basic exploration, preprocessing, model training, evaluation, and saving the final model.

## Business Problem
Healthcare teams often need faster ways to identify patients who may be at higher risk of diabetes. A machine learning model can help flag potential risk earlier so medical professionals can prioritize follow-up screening.

## Dataset
This project uses a small diabetes-style dataset with these fields:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome

Target variable:

- `0` = No diabetes
- `1` = Diabetes

## Tools & Technologies
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- joblib

## Machine Learning Workflow
1. Load the dataset
2. Inspect data quality
3. Separate features and target
4. Split the data into training and test sets
5. Train a Random Forest classifier
6. Evaluate performance with accuracy, precision, recall, F1-score, and confusion matrix
7. Save the trained model

## How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/diabetes-risk-ml-project.git
cd diabetes-risk-ml-project
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the environment
Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the project
```bash
python train_model.py
```

## Project Files
```text
diabetes-risk-ml-project/
│
├── data/
│   └── diabetes.csv
│
├── models/
│   └── diabetes_model.pkl
│
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Key Takeaways
- Built a complete classification pipeline using Python and scikit-learn
- Practiced training and evaluating a supervised learning model
- Used classification metrics to understand model performance
- Saved the trained model for future deployment or API integration

## Future Improvements
- Tune hyperparameters with GridSearchCV
- Compare Logistic Regression, Decision Tree, and XGBoost
- Add feature importance visualization
- Deploy the model with FastAPI
- Create a simple web app interface
