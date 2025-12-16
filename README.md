# Predicting Loan Payback - Kaggle Competition

![Kaggle Challenge](./assets/challenge.png)

This project is part of the **Kaggle Playground Series (Season 5, Episode 11)** competition, focusing on predicting whether a borrower will repay their loan based on various financial and demographic features.

## 📊 Project Overview

The goal is to build a machine learning model that predicts loan repayment likelihood using features such as:
- Credit score
- Annual income
- Debt-to-income ratio
- Loan amount and interest rate
- Demographic information (gender, marital status, education, employment)
- Loan purpose and grade/subgrade

### Evaluation Metric

The competition uses **AUC-ROC (Area Under the ROC Curve)** as the primary evaluation metric. This metric is particularly suitable for binary classification problems with imbalanced datasets, as it measures the model's ability to distinguish between positive and negative classes across various probability thresholds.

![AUC-ROC](./assets/AUC-ROC.png)

## 📁 Project Structure

```
Predicting-Loan-Payback-Kaggle-Competition/
├── data/
│   ├── train.csv          # Training dataset (593,994 samples)
│   └── test.csv           # Test dataset (254,569 samples)
├── models/
│   ├── best_catboost_model.joblib
│   ├── best_lgbm_model.joblib
│   ├── best_logreg_model.joblib
│   ├── best_naive_bayes_model.joblib
│   ├── best_rf_model.joblib
│   ├── best_tree_model.joblib
│   └── best_xgboost_model.joblib
├── assets/
│   ├── AUC-ROC.png
│   └── challenge.png
├── predecting_load_payback_kaggle_competition.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## 📋 Dataset Description

### Features

#### Numerical Features (5)
- **annual_income** (float64): Borrower's yearly income
- **debt_to_income_ratio** (float64): Ratio of borrower's debt to their income (lower is better)
- **credit_score** (int64): Credit bureau score (300-579: Poor, 580-669: Fair, 670-739: Good, 740+: Excellent)
- **loan_amount** (float64): Amount of loan taken
- **interest_rate** (float64): Annual interest rate (%)

#### Categorical Features (6)
- **gender**: Borrower's gender (Male/Female)
- **marital_status**: Marital status (Single, Married, Divorced)
- **education_level**: Education level (High School, Bachelor's, Master's, PhD)
- **employment_status**: Current employment type (Employed, Self-Employed, Unemployed)
- **loan_purpose**: Purpose of the loan (Car, Education, Home, Medical, Debt consolidation, Other)
- **grade_subgrade**: Risk category assigned to loan (A1-G5)

#### Target Variable
- **loan_paid_back** (float64): Binary target variable
  - **1.0**: Borrower paid loan in full
  - **0.0**: Borrower defaulted (did not repay fully)

### Class Distribution
- **Paid Back (1.0)**: ~80% of loans
- **Defaulted (0.0)**: ~20% of loans

The dataset shows a moderate class imbalance, making AUC-ROC a more appropriate metric than simple accuracy.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Predicting-Loan-Payback-Kaggle-Competition.git
cd Predicting-Loan-Payback-Kaggle-Competition
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook predecting_load_payback_kaggle_competition.ipynb
```

2. Run the cells sequentially to:
   - Load and explore the data
   - Perform exploratory data analysis (EDA)
   - Preprocess features
   - Train multiple machine learning models
   - Evaluate model performance
   - Generate predictions

## 🤖 Models Implemented

The project implements and compares multiple machine learning algorithms:

1. **LightGBM Classifier** - Gradient boosting framework
2. **XGBoost Classifier** - Extreme gradient boosting
3. **CatBoost Classifier** - Gradient boosting with categorical features support
4. **Random Forest Classifier** - Ensemble of decision trees
5. **Logistic Regression** - Linear classification model
6. **Decision Tree Classifier** - Single decision tree
7. **Naive Bayes Classifier** - Probabilistic classifier

All models are optimized using **RandomizedSearchCV** with **Stratified K-Fold cross-validation** to handle the class imbalance.

## 📈 Model Training Approach

1. **Data Preprocessing**:
   - Label encoding for categorical features
   - Standard scaling for numerical features
   - Feature engineering as needed

2. **Model Selection**:
   - Stratified K-Fold cross-validation (5 folds)
   - RandomizedSearchCV for hyperparameter tuning
   - AUC-ROC as the scoring metric

3. **Model Persistence**:
   - Best models saved using joblib
   - Stored in the `models/` directory

## 🔍 Key Insights

- The dataset contains **593,994 training samples** and **254,569 test samples**
- No missing values in the dataset
- Moderate class imbalance (80/20 split)
- Multiple categorical features require proper encoding
- Credit score and debt-to-income ratio are likely strong predictors

## 📊 Results

Model performance metrics and comparison can be found in the Jupyter notebook. The best-performing models are saved in the `models/` directory for future predictions.

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and tools
- **LightGBM** - Gradient boosting framework
- **XGBoost** - Extreme gradient boosting
- **CatBoost** - Gradient boosting (implied from saved models)
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment

## 📝 License

This project is created for educational purposes as part of the Kaggle Playground Series competition.

## 🤝 Contributing

This is a competition project, but suggestions and improvements are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or feedback, please open an issue in the repository.

## 🙏 Acknowledgments

- Kaggle for hosting the Playground Series competition
- The machine learning community for various resources and inspiration

---

**Note**: The CSV files in the `data/` directory are ignored in version control as specified in `.gitignore`. Download the dataset from the Kaggle competition page.