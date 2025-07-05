Developing Explainable AI Models for Credit Scoring

This project aims to build a transparent and efficient machine learning model to predict customer creditworthiness. Using logistic regression, the model classifies loan applications as **Good (0)** or **Bad (1)** credit risk, supporting financial institutions in making fair and data-driven decisions.

Project Highlights

- Built a predictive credit scoring model using **Logistic Regression**
- Applied data preprocessing: null value treatment, feature scaling, and label extraction
- Generated outcome probability for each prediction
- Exported trained model and scaler using `joblib` for future integration
- Visualized prediction trends using `matplotlib`
- Created a final result file with actual vs. predicted outcomes and probabilities

Technologies Used

| Category              | Tools & Libraries                                  |
|-----------------------|----------------------------------------------------|
| Programming Language  | Python                                             |
| Libraries             | pandas, numpy, scikit-learn, matplotlib            |
| Data Handling         | MS Excel                                           |
| Model Export          | joblib                                             |
| Development Platform  | Google Colab                                       |

Dataset Description

- Format: Excel file (`.xlsx`)
- Columns:
  - `ID`: Unique Customer Identifier (dropped during processing)
  - Features: Various numeric indicators
  - `TARGET`: Credit outcome (0 = Good loan, 1 = Bad loan)
- Missing values handled using **mean imputation**

Workflow

1. Data Loading & Cleaning
   - Removed irrelevant columns
   - Handled missing values
2. Feature Engineering
   - Extracted labels (`TARGET`)
   - Standardized features using `StandardScaler`
3. Model Training
   - Used **Logistic Regression** from `scikit-learn`
   - 80:20 Train-Test split with stratification
4. Model Evaluation
   - Accuracy, Confusion Matrix, Classification Report
   - Predicted probability for each class
5. Export & Output
   - Model and scaler saved using `joblib`
   - Final prediction results exported as `.csv`
6. Visualization
   - Basic plot showing actual outcomes using `matplotlib`

Output Preview

The model output file (`Model_Prediction.xlsx`) includes:
- `Actual Outcome` (Ground truth)
- `prob_0`: Probability of Good loan
- `prob_1`: Probability of Bad loan
- `predicted_TARGET`: Model-predicted outcome

Sample Code Snippet

```python
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
joblib.dump(classifier, 'Classifier_CreditScoring')

predictions = classifier.predict_proba(X_test)
df_result = pd.concat([
    pd.DataFrame(y_test, columns=['Actual Outcome']),
    pd.DataFrame(predictions, columns=['prob_0', 'prob_1']),
    pd.DataFrame(classifier.predict(X_test), columns=['predicted_TARGET'])
], axis=1)
df_result.to_csv('Model_Prediction.csv')

