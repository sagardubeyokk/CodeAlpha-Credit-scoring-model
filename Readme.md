# Credit Scoring Model
 
A machine learning project that predicts whether a person is likely to become seriously delinquent on a loan within 2 years, using past financial data.
 
## Objective
 
Predict an individual's creditworthiness using classification algorithms (Logistic Regression, Decision Tree, Random Forest), and evaluate the models with Precision, Recall, F1-Score and ROC-AUC.
 
## Dataset
 
- **Source:** [Give Me Some Credit (Kaggle)](https://www.kaggle.com/competitions/GiveMeSomeCredit/data), training file `cs-training.csv` (saved as `credit_data.csv` in this project)
- **Size:** 150,000 customers, 10 input features, 1 target (`SeriousDlqin2yrs`: 1 = defaulted, 0 = repaid)
- **Class imbalance:** 93.3% good borrowers, 6.7% defaulters
- **Missing values:** `MonthlyIncome` (about 20%), `NumberOfDependents` (about 2.6%)

## Approach
 
1. Explored the data (shape, data types, missing values, class balance)
2. Dropped the row-number column `Unnamed: 0`
3. Feature engineering (4 new features):
   - `TotalLatePayments`: sum of the three late-payment counts (each capped at 10)
   - `HasLatePayment`: 1 if the person was ever late, else 0
   - `IncomePerPerson`: `MonthlyIncome / (NumberOfDependents + 1)`
   - `IncomeMissing`: 1 if income was blank, else 0
4. Stratified 80/20 train/test split (`random_state=42`)
5. Preprocessing pipeline: median imputer, then StandardScaler (fit on the training data only, to avoid data leakage)
6. Trained 3 classifiers with balanced class weights: Logistic Regression, Decision Tree, Random Forest
7. Evaluated on the test set with Precision, Recall, F1-Score and ROC-AUC (class 1 = defaulter)
## Results (test set, %)
 
| Model               | Precision | Recall | F1   | ROC-AUC |
|---------------------|-----------|--------|------|---------|
| Logistic Regression | 23.3      | 67.1   | 34.6 | ~83     |
| Decision Tree       | 18.9      | 80.3   | 30.6 | 85.4    |
| **Random Forest**   | 22.9      | 76.1   | 35.2 | **86.7** |
 
All scores are for the defaulter class (class 1).
 
## Conclusion
 
Random Forest gave the best overall performance: the highest ROC-AUC and F1, with Recall close to the Decision Tree's. It was selected as the final model.
 
Accuracy was not used to judge the models, because a model that predicts "good borrower" for everyone already scores 93.3%.
 
Feature importance showed that credit utilisation and payment-history features were the strongest predictors. Two of the four engineered features (`TotalLatePayments` and `HasLatePayment`) ranked in the top three.
 
## How to use the saved model
 
The file `credit_scoring_model.pkl` contains the full pipeline (imputer, scaler and Random Forest). New data must first go through `add_features`, which is saved in `predict_helpers.py`.
 
```python
import pickle
import pandas as pd
from predict_helpers import add_features, feature_cols
 
with open('credit_scoring_model.pkl', 'rb') as f:
    model = pickle.load(f)
 
customer = {
    'RevolvingUtilizationOfUnsecuredLines': 0.95,
    'age': 28,
    'NumberOfTime30-59DaysPastDueNotWorse': 2,
    'DebtRatio': 0.8,
    'MonthlyIncome': 2500,
    'NumberOfOpenCreditLinesAndLoans': 6,
    'NumberOfTimes90DaysLate': 2,
    'NumberRealEstateLoansOrLines': 0,
    'NumberOfTime60-89DaysPastDueNotWorse': 1,
    'NumberOfDependents': 3,
}
 
df = add_features(pd.DataFrame([customer]))[feature_cols]
risk_score = model.predict_proba(df)[0, 1]
print(round(risk_score, 3))
```
 
A higher score means a higher risk. Use the model with the same scikit-learn version listed in `requirements.txt`. Only load `.pkl` files from sources you trust, because pickle files can run code when opened.
 
## Limitations
 
- Precision is low (about 23%): only about 1 in 4 flagged customers is a real defaulter, so flagged cases need human review.
- The scores are risk rankings, not exact default probabilities, because balanced class weights were used.
- The models were not hyperparameter-tuned, and the results come from a single train/test split.
- A real lending decision would also depend on the loan amount, bank policy and legal rules.
## Files
 
- `Loan.ipynb`: full notebook (data exploration, feature engineering, training, evaluation)
- `credit_scoring_model.pkl`: saved pipeline (preprocessor and Random Forest)
- `predict_helpers.py`: `add_features`, `late_cols` and `feature_cols` needed to prepare new data
- `requirements.txt`: library versions

# Thank You !