# Project notes for the student

## What the model is doing
- The input variables are nine numeric water measurements.
- The output is the dataset's `Potability` class (0 or 1).
- This is a binary classification problem.

## Why these algorithms?
- Logistic Regression: simple baseline classifier; scaling is used because feature ranges differ substantially.
- Decision Tree: learns rule-like splits and is relatively easy to interpret.
- Random Forest: combines many decision trees to reduce variance and capture non-linear relationships.

## Why median imputation?
Some numeric records are missing values. The median is less affected by extreme values than the mean. The imputer is part of each pipeline and is fitted on training data only.

## Why stratified train/test split?
The target classes are uneven. Stratification keeps approximately the same class proportions in both subsets.

## Metrics
- Accuracy: fraction of all predictions that are correct.
- Precision: among predictions for a class, how many are correct.
- Recall: among actual examples of a class, how many are found.
- Macro F1: calculates F1 for each class and averages them equally.
- ROC-AUC: measures ranking ability across classification thresholds.
- Recall for class 0 is reported separately because missing a non-potable label can be an important error in this benchmark context.

## Limitations to say clearly
The public benchmark does not establish that its labels are validated under India's drinking-water standard or representative of local groundwater. High test performance only measures agreement with this dataset's labels. Do not claim the app certifies real water safety.
