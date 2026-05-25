import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

# Mock Dataset Creation
# Features: amount, hour_of_day, transaction_velocity, location_risk_score
data = {
    'amount': [100, 50000, 200, 10, 80000, 150, 300, 90000],
    'hour': [14, 3, 15, 12, 2, 16, 10, 1],
    'velocity': [1, 10, 1, 1, 15, 2, 1, 12], # transactions in last hour
    'is_fraud': [0, 1, 0, 0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# SMOTE to balance the data
sm = SMOTE(random_state=42, k_neighbors=2) 
X_res, y_res = sm.fit_resample(X, y)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_res, y_res)

# Save the model
with open('upi_model.pkl', 'wb') as f:
    pickle.dump(model, f)