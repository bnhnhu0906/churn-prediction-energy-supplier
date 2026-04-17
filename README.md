# Churn Prediction for Dutch Energy Supplier

## Overview
Predicted customer churn for a Dutch energy supplier facing a 49% annual 
churn rate, using machine learning on 20,000 customer records.

## Key Results
- Best model: SVM (Radial) — Gini 0.6428, Top-decile lift 1.874
- Strongest predictor: contract flexibility (odds ratio 6.9x)

## Models Compared
- Logistic Regression (Baseline)
- Stepwise Logistic Regression
- CART Decision Tree
- Random Forest
- Support Vector Machine
- Neural Network

## Tools
R · caret · glmnet · e1071 · ggplot2

## Key Finding
High-usage customers were the most likely to churn due to price 
sensitivity, inverting the initial hypothesis of loyalty through consumption.
