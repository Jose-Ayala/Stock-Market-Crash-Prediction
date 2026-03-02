# Stock Market Crash Prediction

## Overview
* The goal of this project was to build a predictive model to identify stocks at risk of a "crash".
* A "crash" is defined as a monthly return < -8%.
* The model was built using data for the years (2019-2023).

## Methodology

### Data Preparation & Predictors
* The target variable, CRASH, was created as a 0/1 variable.
* All predictors were lagged by one month to ensure the model was truly predictive.
* The predictors used were:
  * RET1 (Lagged Return)
  * mcap1 (Lagged Market Cap)
  * volatility1 (Lagged Absolute Return/Volatility)
  * turnover1 (Lagged Turnover)

### Model Exploration
* The data was split into a 70% Training set and a 30% Testing set using the caret package's createDataPartition function.
* Three models were trained and tested:
  * Generalized Linear Model (GLM)
  * Random Forest (RF)
  * Decision Tree (rpart)

## Results
* The models were evaluated by comparing their Area Under the Curve (AUC) on the unseen Test set.
* Generalized Linear Model (GLM) AUC: 0.6654
* Random Forest (RF) AUC: 0.6375
* Decision Tree (Tree) AUC: 0.6227

### Conclusion
* The best-performing model was the Generalized Linear Model (GLM), which had an AUC of 0.6654.
* The GLM function in the caret package was used to estimate a logistic regression.
* This model is designed to estimate the probability of an event occurring and is the best choice because our target variable, CRASH, is a binary variable (0 or 1).
* The GLM model performed better than the more complex Random Forest model.
* The Random Forest model may have been overfitting the training data, while the simpler GLM appeared to be better at generalizing and identifying the true underlying pattern.

## How to Run
1. Ensure R is installed with the following libraries: dplyr, haven, ggplot2, caret, pROC, randomForest.

2. The dataset mret7023 – provides monthly stock price data over the 1970–2023. It is derived from CRSP (https://www.crsp.org/research/crsp-us-stock-databases/).
