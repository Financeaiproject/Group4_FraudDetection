import pandas as pd

# Load data
df = pd.read_csv('../data/PS_20174392719_1491204439457_log.csv')
print("Data loaded! Shape:", df.shape)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Filter only fraud-prone transactions
df_model = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
print("\nAfter filtering shape:", df_model.shape)

# Encode transaction type
df_model['type_encoded'] = (df_model['type'] == 'TRANSFER').astype(int)

# Select features and target
X = df_model[['type_encoded', 'amount', 'oldbalanceOrg',
              'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = df_model['isFraud']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
print("\nPreprocessing complete!")