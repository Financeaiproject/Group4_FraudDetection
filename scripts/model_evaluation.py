import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('../data/PS_20174392719_1491204439457_log.csv')

# Filter
df_model = df[df['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
df_model['type_encoded'] = (df_model['type'] == 'TRANSFER').astype(int)

# Features and target
X = df_model[['type_encoded', 'amount', 'oldbalanceOrg',
              'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = df_model['isFraud']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('Confusion Matrix')
plt.savefig('../outputs/confusion_matrix.png')
plt.show()
print("Evaluation complete!")