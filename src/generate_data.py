"""Generates the synthetic dataset (seeded, reproducible). Synthetic data: not for fraud claims."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 1200
cities = ["Delhi","Mumbai","Bengaluru","Chennai","Kolkata","Hyderabad","Pune","Jaipur"]
city_p = [.18,.20,.16,.11,.09,.11,.08,.07]
channels = ["ATM","Mobile App","Net Banking","Branch","POS","UPI"]
ch_p = [.12,.24,.14,.08,.17,.25]
acct = ["Savings","Current","Salary","Business"]
ac_p = [.45,.2,.2,.15]
ttype = rng.choice(["Credit","Debit"], n, p=[.42,.58])
account_type = rng.choice(acct, n, p=ac_p)
channel = rng.choice(channels, n, p=ch_p)
city = rng.choice(cities, n, p=city_p)
base = rng.lognormal(mean=7.6, sigma=1.0, size=n)
mult = pd.Series(account_type).map({"Savings":1,"Current":1.8,"Salary":1.2,"Business":2.6}).values
amt = base*mult
amt[channel=="ATM"] = np.minimum(amt[channel=="ATM"], 40000)
amt[channel=="UPI"] = np.minimum(amt[channel=="UPI"], 100000)
big = rng.choice(n, 12, replace=False)
amt[big] = rng.uniform(250000, 900000, 12)
amt = np.round(amt, 2)
risk = 0.02 + 0.10*(amt>150000) + 0.03*(channel=="Net Banking") + 0.02*(channel=="Mobile App")
flagged = np.where(rng.random(n) < risk, "Yes", "No")
dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 300*24*60, n), unit="m")
df = pd.DataFrame({
 "Transaction_ID": [f"TXN{100000+i}" for i in range(n)],
 "Transaction_Date": dates.strftime("%Y-%m-%d %H:%M"),
 "Account_ID": [f"ACC{rng.integers(1000,1400)}" for _ in range(n)],
 "Account_Type": account_type, "Transaction_Type": ttype,
 "Channel": channel, "City": city, "Amount": amt, "Flagged": flagged})
df = df.sort_values("Transaction_Date").reset_index(drop=True)
df.loc[rng.choice(n, 24, replace=False), "Amount"] = np.nan
dups = df.sample(8, random_state=1)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
from pathlib import Path
out = Path(__file__).resolve().parent.parent / "data" / "10_bank_transactions.csv"
df.to_csv(out, index=False)
print(df.shape); print(df.head()); print(df.isna().sum()); print(df.Transaction_ID.duplicated().sum()); print(df.Flagged.value_counts())
