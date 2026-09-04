import pandas as pd

cols = ["step","type","amount","oldbalanceOrg","newbalanceOrig","nameDest","oldbalanceDest","newbalanceDest","isFraud","isFlaggedFraud"]
df = pd.read_csv(r"C:\Users\Ishan\Desktop\razorpay\PS_20174392719_1491204439457_log.csv", usecols=cols)

print("rows:", len(df))
print("fraud rate:", df["isFraud"].mean())
print("fraud count:", df["isFraud"].sum())
print()
print("type value counts:")
print(df["type"].value_counts())
print()
print("fraud by type:")
print(df.groupby("type")["isFraud"].sum())
print()
print("isFlaggedFraud sum:", df["isFlaggedFraud"].sum())
print()
print("step range:", df["step"].min(), df["step"].max())
print()
print("fraud rate in last 20% of steps vs first 80%:")
cutoff = df["step"].quantile(0.8)
print("cutoff step:", cutoff)
print("train fraud rate:", df[df["step"] <= cutoff]["isFraud"].mean(), "n=", (df["step"]<=cutoff).sum())
print("test fraud rate:", df[df["step"] > cutoff]["isFraud"].mean(), "n=", (df["step"]>cutoff).sum())
