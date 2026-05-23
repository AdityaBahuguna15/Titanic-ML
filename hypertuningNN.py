import optuna
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler # feature scaling
from sklearn.model_selection import train_test_split # data splitting

root_dir = r"C:\Users\Aditya\ML_Projects\Project 1\titanic"

train_file = 'train.csv'
test_file = 'test.csv'

train_path = os.path.join(root_dir, train_file)
test_path = os.path.join(root_dir, test_file)

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

y = train_data['Survived']
x = train_data.copy()
#print("After copy:", x.columns.tolist())

# Save these BEFORE dropping — needed for test.csv preprocessing later
age_median = x['Age'].median()
fare_median = x['Fare'].median()
embarked_mode = x['Embarked'].mode()[0]
age_bins = [0, 12, 18, 35, 60, 100]

# Feature Engineering:
x['Title'] = train_data['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
x['Title'] = x['Title'].map({
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 
    'Master': 3, 'Dr': 4, 'Rev': 4,
    'Col': 4, 'Major': 4, 'Mlle': 1
}).fillna(4)

x['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
x['IsAlone'] = (x['FamilySize'] == 1).astype(int)

x['Age'] = x['Age'].fillna(x['Age'].median())
x['Fare'] = x['Fare'].fillna(x['Fare'].median())
x['Embarked'] = x['Embarked'].fillna(x['Embarked'].mode()[0])

x['AgeBin'] = pd.cut(x['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0,1,2,3,4])
x['AgeBin'] = x['AgeBin'].astype(int)

x['FareBin'] = pd.qcut(x['Fare'], q=4, labels=[0,1,2,3])
x['FareBin'] = x['FareBin'].astype(int)

x['Sex'] = x['Sex'].map({'male': 0, 'female': 1})
x = pd.get_dummies(x, columns=['Embarked'], drop_first=True)

x = x.drop([
    'PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin',
    'SibSp', 'Parch',    # replaced by FamilySize, IsAlone
    'Age', 'Fare'        # replaced by AgeBin, FareBin
], axis=1)

n_samples, n_features = x.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42)

# Scale
sc = StandardScaler() # Scale our features, feature to have zero mean and unit variance
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# ^ x goes from df to numpy array

#convert to torch tensors
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.to_numpy().astype(np.float32)) # Convert to numpy array
y_test = torch.from_numpy(y_test.to_numpy().astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1) 
y_test = y_test.view(y_test.shape[0], 1) 

def objective(trial):
    # Optuna suggests hyperparameter values to try
    hidden_size1 = trial.suggest_categorical('hidden_size1', [16, 32, 64, 128])
    hidden_size2 = trial.suggest_categorical('hidden_size2', [8, 16, 32, 64])
    dropout1 = trial.suggest_float('dropout1', 0.1, 0.5)
    dropout2 = trial.suggest_float('dropout2', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    step_size = trial.suggest_categorical('step_size', [25, 50, 100])
    gamma = trial.suggest_float('gamma', 0.1, 0.9)

    # Build model with suggested values
    model = nn.Sequential(
        nn.Linear(n_features, hidden_size1),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(hidden_size1, hidden_size2),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(hidden_size2, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5  # in terms of 10-epoch checks

    for epoch in range(300):
        model.train()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Check val loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_test)
                val_loss = criterion(val_pred, y_test).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            model.train()

    return best_val_loss  # Optuna minimizes this


# Run the study
study = optuna.create_study(direction='minimize')  # minimize val_loss
study.optimize(objective, n_trials=100)            # try 100 combinations

# Results
print(f'\nBest val_loss: {study.best_value:.4f}')
print(f'Best hyperparameters: {study.best_params}')

# Train final model with best params
best = study.best_params
final_model = nn.Sequential(
    nn.Linear(n_features, best['hidden_size1']),
    nn.ReLU(),
    nn.Dropout(best['dropout1']),
    nn.Linear(best['hidden_size1'], best['hidden_size2']),
    nn.ReLU(),
    nn.Dropout(best['dropout2']),
    nn.Linear(best['hidden_size2'], 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(final_model.parameters(), lr=best['learning_rate'])
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=best['step_size'], gamma=best['gamma']
)

for epoch in range(300):
    final_model.train()
    y_pred = final_model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

# Evaluate
final_model.eval()
with torch.no_grad():
    y_pred = final_model(x_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Tuned model accuracy: {acc:.4f}')


# ---- Retrain final model on ALL training data ----
# Need to reconstruct x_full from the full x before the train/test split
# Scale using the same scaler
x_full = sc.transform(x)  # x is still the full preprocessed DataFrame
x_full = torch.from_numpy(x_full.astype(np.float32))
y_full = torch.from_numpy(y.to_numpy().astype(np.float32)).view(-1, 1)

# Rebuild model with best params
full_model = nn.Sequential(
    nn.Linear(n_features, best['hidden_size1']),
    nn.ReLU(),
    nn.Dropout(best['dropout1']),
    nn.Linear(best['hidden_size1'], best['hidden_size2']),
    nn.ReLU(),
    nn.Dropout(best['dropout2']),
    nn.Linear(best['hidden_size2'], 1),
    nn.Sigmoid()
)

criterion_full = nn.BCELoss()
optimizer_full = torch.optim.Adam(full_model.parameters(), lr=best['learning_rate'])
scheduler_full = torch.optim.lr_scheduler.StepLR(
    optimizer_full, step_size=best['step_size'], gamma=best['gamma']
)

for epoch in range(300):
    full_model.train()
    y_pred_full = full_model(x_full)
    loss_full = criterion_full(y_pred_full, y_full)
    loss_full.backward()
    optimizer_full.step()
    optimizer_full.zero_grad()
    scheduler_full.step()

print('Retrained on full dataset')

# ---- Preprocess test.csv — identical steps ----
test_data['Title'] = test_data['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Title'].map({
    'Mr': 0, 'Miss': 1, 'Mrs': 2,
    'Master': 3, 'Dr': 4, 'Rev': 4,
    'Col': 4, 'Major': 4, 'Mlle': 1
}).fillna(4)

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

# Use saved statistics from training — never recompute from test data
test_data['Age'] = test_data['Age'].fillna(age_median)
test_data['Fare'] = test_data['Fare'].fillna(fare_median)
test_data['Embarked'] = test_data['Embarked'].fillna(embarked_mode)

test_data['AgeBin'] = pd.cut(test_data['Age'], bins=age_bins, labels=[0,1,2,3,4]).astype(int)
test_data['FareBin'] = pd.qcut(test_data['Fare'], q=4, labels=[0,1,2,3]).astype(int)

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

# Save PassengerId before dropping
passenger_ids = test_data['PassengerId']

test_data = test_data.drop([
    'PassengerId', 'Name', 'Ticket', 'Cabin',
    'SibSp', 'Parch', 'Age', 'Fare'
], axis=1)

# ---- Verify columns match ----
print("Train columns:", x.columns.tolist())
print("Test columns: ", test_data.columns.tolist())
print("Match:", x.columns.tolist() == test_data.columns.tolist())

# ---- Scale using same scaler ----
x_kaggle = sc.transform(test_data)
x_kaggle = torch.from_numpy(x_kaggle.astype(np.float32))

# ---- Generate predictions ----
full_model.eval()
with torch.no_grad():
    kaggle_preds = full_model(x_kaggle)
    kaggle_preds_cls = kaggle_preds.round().int().squeeze()

# ---- Save submission ----
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': kaggle_preds_cls.numpy()
})

submission_path = os.path.join(root_dir, 'submission.csv')
submission.to_csv(submission_path, index=False)
print(f'\nSubmission saved to {submission_path}')
print(submission.head(10))
print(f'Total predictions: {len(submission)}')
print(f'Predicted survived: {kaggle_preds_cls.sum().item()} ({kaggle_preds_cls.float().mean().item()*100:.1f}%)')