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
# 1) Title Extractionfrom Name — Mr, Mrs, Miss, Master correlates strongly with survival
x['Title'] = train_data['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
x['Title'] = x['Title'].map({
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 
    'Master': 3, 'Dr': 4, 'Rev': 4,
    'Col': 4, 'Major': 4, 'Mlle': 1
}).fillna(4)
#print("After title:", x.columns.tolist())

# 2) Family Size — being alone vs large family affected survival
x['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
x['IsAlone'] = (x['FamilySize'] == 1).astype(int)
#print("After family:", x.columns.tolist())

# Preprocessing: Fill out missing values:
x['Age'] = x['Age'].fillna(x['Age'].median())
x['Fare'] = x['Fare'].fillna(x['Fare'].median())
x['Embarked'] = x['Embarked'].fillna(x['Embarked'].mode()[0])
#print("After fillna:", x.columns.tolist())

# 3) Age bins — child vs adult vs elderly matters more than exact age
x['AgeBin'] = pd.cut(x['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0,1,2,3,4])
x['AgeBin'] = x['AgeBin'].astype(int)

# 4) Fare bins — same idea
x['FareBin'] = pd.qcut(x['Fare'], q=4, labels=[0,1,2,3])
x['FareBin'] = x['FareBin'].astype(int)

# ---- Step 4: Encode categoricals ----
x['Sex'] = x['Sex'].map({'male': 0, 'female': 1})
x = pd.get_dummies(x, columns=['Embarked'], drop_first=True)

# ---- Step 5: Drop everything no longer needed ----
x = x.drop([
    'PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin',
    'SibSp', 'Parch',    # replaced by FamilySize, IsAlone
    'Age', 'Fare'        # replaced by AgeBin, FareBin
], axis=1)

# ^ Currently x is a pandas df

# ---- Step 6: Sanity check ----
# print(x.columns.tolist())
# print(x.shape)
# print(x.isnull().sum())

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

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) # 7 inputs, 1 output
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x)) # Sigmoid fnct returns a value btwn 0-1
        return y_pred

class ImprovedModel(nn.Module): # 1-2% increase in accuracy
    def __init__(self, n_input_features):
        super(ImprovedModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input_features, 32),
            nn.ReLU(),
            nn.Dropout(0.4),      # prevents overfitting
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

model = ImprovedModel(n_features)
# model = LogisticRegression(n_features)
# 2) Loss and Optimizer
learning_rate = 0.02
criterion = nn.BCELoss() # Classes: only 2, eg: Outcomes: Yes/No, 0/1
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

best_val_loss = float('inf')
patience = 50        # how many epochs to wait for improvement
patience_counter = 0
best_model_state = None

# 3) Training loop
num_epochs = 400
for epoch in range(num_epochs):
    model.train()
    #Forward pass
    y_pred = model(x_train)
    loss = criterion(y_pred,y_train)

    #Backward pass
    loss.backward()

    #Update
    optimizer.step()

    #Zero grad
    optimizer.zero_grad()
    scheduler.step()

    # if(epoch+1) % 10 == 0:
    #     print(f'epoch{epoch+1}, loss = {loss.item():.4f}')
    # Check both losses every 10 epochs
    if (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(x_test)
            val_loss = criterion(val_pred, y_test)
        print(f'epoch {epoch+1}, train_loss = {loss.item():.4f}, val_loss = {val_loss.item():.4f}')
        
        # Check if val loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # save best weights
            print(f'  → New best val_loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'  → No improvement ({patience_counter}/{patience//10} strikes)')

        if patience_counter >= patience // 10:
            print(f'Early stopping at epoch {epoch+1}')
            break

        
        model.train()

# Restore best weights before evaluation
model.load_state_dict(best_model_state)
print(f'Restored best model with val_loss: {best_val_loss:.4f}')

#Evaluation:
with torch.no_grad():
    y_pred = model(x_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0]) # for every pred correct: +1
    print(f'accuracy = {acc:.4f}')


# # Preprocessing to test.csv
# # Preprocessing test.csv — apply same steps in same order
# test_data['Title'] = test_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# test_data['Title'] = test_data['Title'].map({
#     'Mr': 0, 'Miss': 1, 'Mrs': 2,
#     'Master': 3, 'Dr': 4, 'Rev': 4,
#     'Col': 4, 'Major': 4, 'Mlle': 1
# }).fillna(4)

# test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
# test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

# # Use saved statistics — NOT x['Age'] which no longer exists
# test_data['Age'] = test_data['Age'].fillna(age_median)
# test_data['Fare'] = test_data['Fare'].fillna(fare_median)
# test_data['Embarked'] = test_data['Embarked'].fillna(embarked_mode)

# test_data['AgeBin'] = pd.cut(test_data['Age'], bins=age_bins, labels=[0,1,2,3,4])
# test_data['AgeBin'] = test_data['AgeBin'].astype(int)

# test_data['FareBin'] = pd.qcut(test_data['Fare'], q=4, labels=[0,1,2,3])
# test_data['FareBin'] = test_data['FareBin'].astype(int)

# test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
# test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

# test_data = test_data.drop([
#     'PassengerId', 'Name', 'Ticket', 'Cabin',
#     'SibSp', 'Parch',
#     'Age', 'Fare'
# ], axis=1)

# # Verify columns match training x exactly
# print("Test columns:", test_data.columns.tolist())
# print("Train columns:", list(x.columns) if hasattr(x, 'columns') else "x already numpy")

# # Keep same columns as training x
# test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# x_kaggle = sc.transform(test_data)

# # Convert to tensor
# x_kaggle = torch.from_numpy(x_kaggle.astype(np.float32))

# # Run predictions
# model.eval()
# with torch.no_grad():
#     kaggle_preds = model(x_kaggle)
#     kaggle_preds_cls = kaggle_preds.round().int().squeeze()

# # Save to CSV for Kaggle submission
# submission = pd.DataFrame({
#     'PassengerId': pd.read_csv(test_path)['PassengerId'],
#     'Survived': kaggle_preds_cls.numpy()
# })

# submission.to_csv(r"C:\Users\Aditya\ML_Projects\Project 1\titanic\submission.csv", index=False)
# print("Submission saved")
# print(submission.head(10))

'''
x['Embarked'] = x['Embarked'].fillna(x['Embarked'].mode()[0])
- .mode() returns the most frequently occurring value — in Titanic that's "S" (Southampton), where most people boarded.
- The [0] is there because .mode() returns a list in case there's a tie — you just grab the first one.

x['Sex'] = x['Sex'].map({'male': 0, 'female': 1})
- This works cleanly for Sex because there are only two categories — it's binary.

x = pd.get_dummies(x, columns=['Embarked'], drop_first=True)
- Embarked has three categories — S, C, Q — so you can't just do 0, 1, 2. That would imply Q is "greater than" C which is 
"greater than" S, which is meaningless. Instead you use one-hot encoding, which is what get_dummies does.

Before:               After get_dummies:
Embarked              Embarked_C    Embarked_Q
S          →          0             0
C          →          1             0
Q          →          0             1
S          →          0             0
drop_first=True drops the first category (S) because it's redundant — if both Embarked_C and Embarked_Q are 0
, you already know the person embarked at S. Keeping it would give the model duplicate information.



Code 1 — Logistic Regression model
Input (30 features) → Linear layer → Sigmoid → Output (0 or 1)

Single linear layer, no hidden layers
Sigmoid squishes output to 0-1 probability
Binary Cross Entropy loss (BCELoss) — two classes only
Solving: binary classification (malignant vs benign)

Code 2 — Feedforward Neural Network
Input (784) → Linear → ReLU → Linear → Output (10 classes)

Two linear layers with a hidden layer in between
ReLU activation adds non-linearity so it can learn complex patterns
CrossEntropyLoss — handles multiple classes
Solving: multiclass classification (digits 0-9)
'''