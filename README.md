# Hypertension Risk Prediction Project


[![Video Link](https://img.shields.io/badge/Video-Demo-red)](https://www.youtube.com/watch?v=NXmLUsdyNLE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/ankushpanday1/hypertension-risk-prediction-dataset)

## 📋 Project Overview

This project addresses the critical challenge of predicting hypertension risk using machine learning techniques. Hypertension (persistently high blood pressure) is a major contributor to cardiovascular diseases, stroke, and kidney failure. Early detection enables timely interventions that can significantly reduce long-term health complications.

The project implements a comprehensive machine learning pipeline featuring advanced feature engineering, class imbalance handling, and ensemble methods to predict hypertension risk from clinical and lifestyle data.

## 👥 Team Members

- **Sohaib Chachar**: Data preprocessing, feature engineering, model development, interactive visualizations, SMOTE implementation, and report writing
- **Thomas Bowen**: Feature engineering, model development, attribute selection, ensembling, code integration, and video recording
- **Alexander Zugaj**: Feature engineering, model development, ensemble comparisons (voting vs stacking vs bagging), ExtraTrees implementation, and video production

## 🎯 Objectives

1. **Enhanced Feature Engineering**: Created nine new derived features beyond the original 23 variables to capture subtle patterns in the data
2. **Comprehensive Model Evaluation**: Comparative study of nine different classification algorithms with rigorous evaluation metrics
3. **Class Imbalance Handling**: Implementation of SMOTE (Synthetic Minority Oversampling Technique) to address imbalanced classes
4. **Ensemble Methods**: Development of stacked ensemble models and comparison of voting, stacking, and bagging approaches
5. **Hyperparameter Optimization**: Randomized Search CV for fine-tuning model parameters
6. **Probability Calibration**: Improved prediction reliability through calibration techniques

## 📊 Dataset

- **Source**: [Hypertension Risk Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/ankushpanday1/hypertension-risk-prediction-dataset)
- **Records**: 174,982 samples
- **Features**: 23 original variables including:
  - **Demographics**: Country, Age, Gender
  - **Lifestyle**: Smoking Status, Alcohol Intake, Physical Activity Level
  - **Clinical Measures**: BMI, Systolic BP, Diastolic BP, Cholesterol, Glucose, Heart Rate
  - **Family History**: Hypertension family history indicators
  - **Health Metrics**: LDL, HDL, Triglycerides, Sleep Duration, Stress Level, Salt Intake

### Class Distribution
- **High Hypertension**: 71.88%
- **Low Hypertension**: 28.12%

## 🔧 Features

### Feature Engineering

Nine new engineered features were created:

1. **Pulse_Pressure** = Systolic_BP - Diastolic_BP
2. **Chol_Ratio** = LDL / (HDL + 1e-6)
3. **Age_BMI** = Age × BMI (interaction term)
4. **Smoking_Status_ord**: Ordinal encoding of smoking status
5. **Physical_Activity_Level_ord**: Ordinal encoding of activity levels
6. **Education_Level_ord**: Ordinal encoding of education
7. **Employment_Status_ord**: Ordinal encoding of employment
8. **Sleep_Cat**: Categorical binning of sleep duration
9. **Alcohol_Quartile**: Quartile-based categorization of alcohol intake

### Interactive Exploratory Analysis

An interactive Plotly + ipywidgets dashboard allows users to:
- Select any two features for visualization
- Choose plot types (Histogram, Box, Violin, Scatter)
- Dynamically explore relationships between features and hypertension status
- Test hypotheses on-the-fly without generating excessive static plots

## 🤖 Models Implemented

The project benchmarks **nine classification algorithms**:

### Linear Models
- **Logistic Regression**: Baseline linear classifier

### Ensemble Methods
- **Random Forest**: 200 trees with hyperparameter tuning
- **Bagging Classifier**: Bootstrap aggregating ensemble
- **AdaBoost**: Adaptive boosting algorithm
- **Gradient Boosting**: Gradient-boosted decision trees

### Advanced Boosters
- **XGBoost**: Extreme gradient boosting
- **LightGBM**: Microsoft's gradient boosting framework
- **CatBoost**: Categorical boosting algorithm

### Neural Networks
- **MLPClassifier**: Multi-layer perceptron with 100-50 hidden units

### Ensemble Techniques
- **Voting Classifier**: Hard and soft voting combinations
- **Stacking Classifier**: Meta-learner with Random Forest
- **ExtraTrees**: Extremely randomized trees

## 📈 Methodology

### Data Preprocessing Pipeline

1. **Feature Transformation**:
   - Numerical features: StandardScaler normalization
   - Categorical features: OneHotEncoder with unknown value handling
   - Combined using sklearn's ColumnTransformer

2. **Train/Test Split**:
   - 80/20 stratified split (maintains class distribution)
   - Training: ~140k samples
   - Testing: ~35k samples

3. **Class Imbalance Handling**:
   - **SMOTE** applied only to training data
   - Balances classes from 71.88/28.12 to 50/50
   - Test set maintains original imbalance for realistic evaluation

4. **Feature Selection**:
   - Mutual Information classification for attribute selection
   - Identifies most informative features for prediction

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve
- **Cross-Validation**: 5-fold stratified CV for robust performance estimation
- **Confusion Matrix**: Detailed classification breakdown

### Model Training

- **Hyperparameter Tuning**: RandomizedSearchCV for optimal parameter selection
- **Probability Calibration**: Improved prediction reliability
- **Cross-Validation**: 5-fold stratified CV to prevent overfitting

## 📦 Installation

### Requirements

```bash
pip install pandas numpy scikit-learn
pip install plotly ipywidgets
pip install seaborn matplotlib
pip install imbalanced-learn
pip install xgboost lightgbm catboost
```

### Key Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning algorithms and preprocessing
- `plotly`: Interactive visualizations
- `ipywidgets`: Interactive widgets for Jupyter notebooks
- `seaborn` & `matplotlib`: Static visualizations
- `imbalanced-learn`: SMOTE implementation
- `xgboost`, `lightgbm`, `catboost`: Advanced boosting algorithms

## 🚀 Usage

### Running the Notebook

1. **Mount Google Drive** (if using Google Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Load the dataset**:
```python
df = pd.read_csv('hypertension_dataset.csv')
```

3. **Run cells sequentially**:
   - Data loading and exploration
   - Feature engineering
   - Interactive visualizations
   - Model training and evaluation
   - Ensemble comparisons

### Interactive Visualization

The notebook includes an interactive dashboard for exploratory data analysis:

```python
# Select features and plot type using dropdown menus
# Features include: Age, BMI, Blood Pressure metrics, Lifestyle factors, etc.
# Plot types: Histogram, Box, Violin, Scatter
```

## 📊 Results

### Model Performance Summary

All models were evaluated using both accuracy and ROC-AUC metrics:

- **Best Cross-Validation Accuracy**: Random Forest (~71.9%)
- **ROC-AUC Performance**: ~0.50 on hold-out test set
- **Key Finding**: Despite comprehensive feature engineering and ensemble methods, the dataset shows limited predictive signal, suggesting highly nonlinear relationships or potential data quality issues

### Important Findings

1. **No Linear Correlations**: Pearson correlations between individual features and target were near zero, indicating nonlinear relationships
2. **Feature Engineering Impact**: Engineered features (Pulse Pressure, Cholesterol Ratio, Age×BMI) captured important interactions
3. **Ensemble Benefits**: Stacked ensembles showed marginal improvements over individual models
4. **SMOTE Effectiveness**: Class rebalancing improved model training stability

## 📚 Literature Review

The project builds upon previous work on the same dataset:

| Author | Model | CV Accuracy | Notes |
|--------|-------|-------------|-------|
| Devraai (2025) | Random Forest | 71.7% | Simple RF, low feature correlations |
| Aolcen (2025) | Deep Neural Network | 71.7% | Dense layers, Adam optimizer |
| Jason Mguni (2025) | Random Forest (tuned) | 71.9% | Hyperparameter search, feature importance |
| Ober (2025) | Logistic Regression + SMOTE | 76.1% | Best CV accuracy |

**Our Contribution**: Comprehensive feature engineering, extensive model comparison, rigorous ROC-AUC evaluation, and ensemble method analysis.

## 🗂️ File Structure

```
.
├── project_complete-1.ipynb    # Main Jupyter notebook with complete analysis
└── README.md                   # This file
```

## 🔬 Key Insights

1. **Feature Engineering**: Creating domain-specific features (Pulse Pressure, Cholesterol Ratio) provides richer information than raw measurements
2. **Class Imbalance**: SMOTE effectively balances training data without leaking information to test set
3. **Model Diversity**: Testing multiple algorithms reveals which approaches work best for this specific problem
4. **Ensemble Learning**: Combining multiple models through stacking or voting can capture complementary patterns
5. **Evaluation Rigor**: Using ROC-AUC alongside accuracy provides deeper insight into model performance

## 🤝 Contributions

Contributions to this project are welcome! Areas for future improvement:

- Additional feature engineering techniques
- Deep learning architectures
- Advanced ensemble methods
- External validation on different datasets
- Feature importance analysis and interpretability



## 🔗 Links

- **Video Presentation**: [YouTube](https://www.youtube.com/watch?v=NXmLUsdyNLE)
- **Dataset**: [Kaggle - Hypertension Risk Prediction Dataset](https://www.kaggle.com/datasets/ankushpanday1/hypertension-risk-prediction-dataset)

---

**Note**: This project demonstrates a comprehensive machine learning pipeline for healthcare prediction tasks. Results highlight the importance of thorough evaluation metrics (ROC-AUC) beyond simple accuracy scores, especially for imbalanced datasets in clinical settings.


