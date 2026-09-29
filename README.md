# Diabetes Classification — Machine Learning Demo

This project demonstrates a supervised machine learning workflow using a diabetes-style dataset. It loads and checks the data, trains a Random Forest classifier, evaluates predictions on a held-out test set, and saves the trained model.

**This is an educational project, not a clinical screening or diagnostic tool.** The repository does not document the dataset’s original source or establish that the model generalizes to real patients.

## Dataset

`data/diabetes.csv` contains 768 rows, eight input features, and a binary `Outcome` column. The features are Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age. The script checks the file for missing values; none were reported in the included dataset.

## Method

- Split the data into training and test sets (80/20), stratified by `Outcome`.
- Train a `RandomForestClassifier` with 150 trees, a maximum depth of 5, and balanced class weights.
- Report accuracy and a classification report, then display a confusion matrix.
- Save the trained model to `models/diabetes_model.pkl`.

## Results

On the included test split of 154 rows, the model achieved **65.6% accuracy**. For the positive (`Outcome = 1`) class, precision was **0.61**, recall was **0.71**, and F1-score was **0.66**.

These are results from one split of a small demo dataset. The model has not been externally validated.

## Run locally

Clone the repository and enter its folder:

```bash
git clone https://github.com/patrickfelder777-cpu/diabetes-risk-ml-project.git
cd diabetes-risk-ml-project
```

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate
```

Then run:

```bash
pip install -r requirements.txt
python train_model.py
```

The script prints the data checks and evaluation results, opens a confusion matrix plot, and creates `models/diabetes_model.pkl`. The saved model file is generated locally and is excluded from Git by `.gitignore`.

## Project files

- `train_model.py` — data loading, training, evaluation, and model saving
- `data/diabetes.csv` — included demo dataset
- `requirements.txt` — Python dependencies
- `models/` — created when the script saves the model

## Next steps

Document the dataset’s provenance, compare against a simple baseline, assess performance across multiple splits, and evaluate with data appropriate to the intended use before considering any real-world application.
