import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

root_dir = r"C:\Users\Aditya\ML_Projects\Project 1\titanic"
train_path = os.path.join(root_dir, 'train.csv')
test_path = os.path.join(root_dir, 'test.csv')

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

y = train_data['Survived']
x = train_data.copy()

# Save statistics before dropping
age_median = x['Age'].median()
fare_median = x['Fare'].median()
embarked_mode = x['Embarked'].mode()[0]

# Feature engineering
x['Title'] = x['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
x['Title'] = x['Title'].map({
    'Mr': 0, 'Miss': 1, 'Mrs': 2,
    'Master': 3, 'Dr': 4, 'Rev': 4,
    'Col': 4, 'Major': 4, 'Mlle': 1
}).fillna(4)

x['FamilySize'] = x['SibSp'] + x['Parch'] + 1
x['IsAlone'] = (x['FamilySize'] == 1).astype(int)

x['Age'] = x['Age'].fillna(age_median)
x['Fare'] = x['Fare'].fillna(fare_median)
x['Embarked'] = x['Embarked'].fillna(embarked_mode)

x['AgeBin'] = pd.cut(x['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
x['FareBin'] = pd.qcut(x['Fare'], q=4, labels=[0,1,2,3]).astype(int)

x['Sex'] = x['Sex'].map({'male': 0, 'female': 1})
x = pd.get_dummies(x, columns=['Embarked'], drop_first=True)

x = x.drop([
    'PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin',
    'SibSp', 'Parch', 'Age', 'Fare'
], axis=1)

print("Features:", x.columns.tolist())
print("Shape:", x.shape)

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}


# Train
# No tuning
# rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Some factors added to help increase accuracy
# rf = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=8,
#     min_samples_split=4,
#     min_samples_leaf=2,
#     max_features='sqrt',
#     class_weight='balanced',
#     random_state=42
# ) # Increased from 82% -> 83%

# rf.fit(x_train, y_train)

# Grid Search: Finds best hyperparameters
grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=5,               # 5-fold cross validation
    scoring='accuracy',
    n_jobs=-1           # use all CPU cores
)

grid_search.fit(x_train, y_train)

print(f'Best params: {grid_search.best_params_}')
print(f'Best CV accuracy: {grid_search.best_score_:.4f}')

# Evaluate best model on test set
best_rf = grid_search.best_estimator_
preds = best_rf.predict(x_test)
print(f'Test accuracy: {accuracy_score(y_test, preds):.4f}')

# # Evaluate
# preds = rf.predict(x_test)
# acc = accuracy_score(y_test, preds)
# print(f'\nRandom Forest accuracy: {acc:.4f}')

# # This breaks down precision and recall per class — more informative than just accuracy
# print('\nClassification Report:')
# print(classification_report(y_test, preds, target_names=['Died', 'Survived']))

# # Feature importance — shows which features the forest relied on most
# importances = pd.Series(rf.feature_importances_, index=x.columns)
# importances = importances.sort_values(ascending=False)
# print('\nFeature Importances:')
# print(importances)

# ---- Retrain best model on ALL training data ----
# During development you used 80% to train and 20% to evaluate
# For final submission, use everything — more data = better model
best_rf.fit(x, y)

# ---- Preprocess test.csv — same steps as train ----
test_data['Title'] = test_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Title'].map({
    'Mr': 0, 'Miss': 1, 'Mrs': 2,
    'Master': 3, 'Dr': 4, 'Rev': 4,
    'Col': 4, 'Major': 4, 'Mlle': 1
}).fillna(4)

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

# Use saved statistics from training data
test_data['Age'] = test_data['Age'].fillna(age_median)
test_data['Fare'] = test_data['Fare'].fillna(fare_median)
test_data['Embarked'] = test_data['Embarked'].fillna(embarked_mode)

test_data['AgeBin'] = pd.cut(test_data['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)
test_data['FareBin'] = pd.qcut(test_data['Fare'], q=4, labels=[0,1,2,3]).astype(int)

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

# Save PassengerId before dropping — needed for submission
passenger_ids = test_data['PassengerId']

test_data = test_data.drop([
    'PassengerId', 'Name', 'Ticket', 'Cabin',
    'SibSp', 'Parch', 'Age', 'Fare'
], axis=1)

# ---- Verify columns match exactly ----
print("Train columns:", x.columns.tolist())
print("Test columns: ", test_data.columns.tolist())
print("Match:", x.columns.tolist() == test_data.columns.tolist())

# ---- Generate predictions ----
kaggle_preds = best_rf.predict(test_data)

# ---- Save submission ----
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': kaggle_preds
})

submission_path = os.path.join(root_dir, 'submission.csv')
submission.to_csv(submission_path, index=False)
print(f'\nSubmission saved to {submission_path}')
print(submission.head(10))
print(f'Total predictions: {len(submission)}')
print(f'Predicted survived: {kaggle_preds.sum()} ({kaggle_preds.mean()*100:.1f}%)')