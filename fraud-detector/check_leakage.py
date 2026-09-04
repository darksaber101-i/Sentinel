import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")
features = joblib.load("models/feature_columns.pkl")

importances = model.feature_importances_
for f, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{f:25s} {imp:.4f}")

print()
print("--- Checking newbalanceOrig for fraud vs non-fraud (sample) ---")
usecols = ["type","amount","oldbalanceOrg","newbalanceOrig","isFraud"]
df = pd.read_csv(r"C:\Users\Ishan\Desktop\razorpay\PS_20174392719_1491204439457_log.csv", usecols=usecols, nrows=2000000)
fraud = df[df.isFraud==1]
legit = df[df.isFraud==0]
print("Fraud newbalanceOrig == 0:", (fraud.newbalanceOrig == 0).mean())
print("Legit newbalanceOrig == 0:", (legit.newbalanceOrig == 0).mean())
print("Fraud oldbalanceOrg == amount (fully drained):", (abs(fraud.oldbalanceOrg - fraud.amount) < 0.01).mean())
