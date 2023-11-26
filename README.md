# Finding Donors for CharityML Project

## Overview

This project aims to help CharityML, a non-profit organization, in identifying potential donors based on the 1994 U.S. Census data. The goal is to build a model that accurately predicts whether an individual makes more than $50,000 annually. Several supervised learning algorithms are explored, and the selected model is optimized to achieve the best performance.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- NumPy, Pandas, Matplotlib, and Scikit-Learn libraries

### Installation

```bash
pip install numpy pandas matplotlib scikit-learn
```

# Project Structure

- **finding_donors.ipynb:** Jupyter Notebook containing the project code.
- **visuals.py:** Python script with visualization functions.
- **README.md:** Project documentation.

## Exploring the Data

The dataset used in this project is sourced from the UCI Machine Learning Repository. The features include demographic information, and the target label is whether an individual earns more than $50,000 annually.

## Data Preprocessing

- Skewed continuous features like 'capital-gain' and 'capital-loss' are log-transformed.
- Numerical features are normalized.
- Categorical variables are one-hot encoded.

## Model Selection

Three supervised learning models are chosen for evaluation:

1. **Logistic Regression:**
   - Real-world application: Cancer detection.
   - Strengths: Fast, provides good results with limited features.
   - Weaknesses: Struggles with complex relationships among features.
   - Good Candidate: Suitable for this problem with cleaned data and fewer features.

2. **Gradient Boosting:**
   - Real-world application: Anomaly detection in highly unbalanced data.
   - Strengths: Corrects errors efficiently through step-by-step learning.
   - Weaknesses: May overfit with noisy data.
   - Good Candidate: Appropriate for structured and cleaned data.

3. **Random Forest:**
   - Real-world application: Multi-class object detection in computer vision.
   - Strengths: Fast, less prone to overfitting.
   - Weaknesses: Slower with a large number of trees.
   - Good Candidate: Effective when dealing with categorical values.

## Model Evaluation

The models are evaluated using accuracy and F-score metrics. A naive predictor is established as a benchmark for comparison.

## Model Tuning

The Gradient Boosting model is selected as the final model and fine-tuned using GridSearchCV to improve its performance.

## Results

The optimized model achieved the following scores on the testing data:

- Accuracy: 87.08%
- F-score: 75.31%

These scores are an improvement over the unoptimized model and significantly outperform the naive predictor benchmarks.

## Feature Importance

The top five features influencing the prediction are identified using the Gradient Boosting classifier:

1. Capital-gain
2. Capital-loss
3. Age
4. Marital status
5. Education-num

These features align with expectations, confirming the importance of financial indicators, age, and marital status.

## Conclusion

The Gradient Boosting model demonstrates superior performance in predicting potential donors. The selected features provide valuable insights for targeted outreach by CharityML.
