"""
Project 10: Banking Transaction Pattern Analysis
------------------------------------------------
Synthetic dataset. Results describe transaction behaviour only and must NOT be
used to make fraud claims. The `Flagged` column is a supplied label, not proof
of fraud.

Run from the project root:
    python src/analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "10_bank_transactions.csv"
TABLES = ROOT / "outputs" / "tables"
OUT = ROOT / "outputs"
TABLES.mkdir(parents=True, exist_ok=True)

THRESHOLD = 100_000  # task 14: selected threshold (INR)
BAND_EDGES = [-np.inf, 1_000, 5_000, 10_000, 50_000, 100_000, np.inf]
BAND_LABELS = ["<=1K", "1K-5K", "5K-10K", "10K-50K", "50K-100K", ">100K"]  # upper edge inclusive

report = []  # lines for the text summary


def say(text=""):
    print(text)
    report.append(str(text))


def section(title):
    say()
    say("=" * 70)
    say(title)
    say("=" * 70)


def save(df, name):
    df.to_csv(TABLES / f"{name}.csv")


def inr(x):
    return f"{x:,.2f}"


# --------------------------------------------------------------------------
# 1. Load and inspect
# --------------------------------------------------------------------------
section("1. LOAD AND INSPECT")
raw = pd.read_csv(DATA)
say(f"Shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
say("\nColumn types:")
say(raw.dtypes.to_string())
say("\nFirst 5 rows:")
say(raw.head().to_string(index=False))
say("\nMissing values per column:")
say(raw.isna().sum().to_string())
say("\nCategory values:")
for col in ["Account_Type", "Transaction_Type", "Channel", "City", "Flagged"]:
    say(f"  {col}: {sorted(raw[col].dropna().unique())}")

# --------------------------------------------------------------------------
# 3. Duplicate Transaction_ID (done before imputation so we do not impute
#    values on rows that are about to be removed)
# --------------------------------------------------------------------------
section("3. DUPLICATE TRANSACTION_ID CHECK")
dup_mask = raw["Transaction_ID"].duplicated(keep=False)
n_dup_ids = raw.loc[dup_mask, "Transaction_ID"].nunique()
n_extra = raw["Transaction_ID"].duplicated(keep="first").sum()
say(f"Transaction_IDs appearing more than once: {n_dup_ids}")
say(f"Extra (repeat) rows: {n_extra}")
exact = raw.duplicated(keep="first").sum()
say(f"Of these, fully identical rows: {exact}")
dups = raw[dup_mask].sort_values("Transaction_ID")
dups.to_csv(TABLES / "03_duplicate_transactions.csv", index=False)
say("Handling: keep the first occurrence of each Transaction_ID, drop repeats.")
df = raw.drop_duplicates(subset="Transaction_ID", keep="first").copy()
say(f"Rows after de-duplication: {len(df)}")

# --------------------------------------------------------------------------
# 2. Missing transaction amounts
# --------------------------------------------------------------------------
section("2. MISSING TRANSACTION AMOUNTS")
n_missing = int(df["Amount"].isna().sum())
say(f"Missing Amount values (after de-duplication): {n_missing} "
    f"({n_missing / len(df):.2%})")
df["Amount_Imputed"] = df["Amount"].isna()
group_median = df.groupby(["Account_Type", "Transaction_Type"])["Amount"].transform("median")
df["Amount"] = df["Amount"].fillna(group_median).fillna(df["Amount"].median())
say("Handling: filled with the median Amount of the same Account_Type and "
    "Transaction_Type (median is robust to the heavy right skew).")
say("Imputed rows are marked in column Amount_Imputed; they are kept so that "
    "transaction counts stay complete.")
say(f"Missing Amount values after handling: {int(df['Amount'].isna().sum())}")
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"])
df.to_csv(OUT / "cleaned_transactions.csv", index=False)

# --------------------------------------------------------------------------
# 4. Total credits and debits
# --------------------------------------------------------------------------
section("4. TOTAL CREDITS AND DEBITS")
cd = df.groupby("Transaction_Type")["Amount"].agg(["count", "sum", "mean"])
cd.columns = ["Transactions", "Total_Amount", "Mean_Amount"]
cd["Share_of_Value_%"] = (cd["Total_Amount"] / cd["Total_Amount"].sum() * 100).round(2)
say(cd.round(2).to_string())
total_credit = cd.loc["Credit", "Total_Amount"]
total_debit = cd.loc["Debit", "Total_Amount"]
net = total_credit - total_debit
say(f"\nTotal credits: {inr(total_credit)}")
say(f"Total debits : {inr(total_debit)}")
say(f"Net (credits - debits): {inr(net)}")
save(cd.round(2), "04_credit_debit_summary")

# --------------------------------------------------------------------------
# 5. Mean, median, std
# --------------------------------------------------------------------------
section("5. AMOUNT STATISTICS")
amt = df["Amount"].to_numpy()
mean_, median_, std_ = amt.mean(), np.median(amt), amt.std(ddof=1)
say(f"Mean   : {inr(mean_)}")
say(f"Median : {inr(median_)}")
say(f"Std dev: {inr(std_)} (sample, ddof=1)")
say(f"Skew note: mean/median = {mean_ / median_:.2f}. A mean well above the "
    "median indicates a right-skewed distribution driven by large transactions.")
pd.DataFrame({"Mean": [mean_], "Median": [median_], "Std_Dev": [std_]}).round(2).to_csv(
    TABLES / "05_amount_statistics.csv", index=False)

# --------------------------------------------------------------------------
# 6. Volume by channel
# --------------------------------------------------------------------------
section("6. TRANSACTION VOLUME BY CHANNEL")
ch = df.groupby("Channel")["Amount"].agg(["count", "sum", "mean", "median"])
ch.columns = ["Transactions", "Total_Value", "Mean_Amount", "Median_Amount"]
ch["Volume_Share_%"] = (ch["Transactions"] / ch["Transactions"].sum() * 100).round(2)
ch = ch.sort_values("Transactions", ascending=False)
say(ch.round(2).to_string())
save(ch.round(2), "06_channel_summary")

# --------------------------------------------------------------------------
# 7. Value by city
# --------------------------------------------------------------------------
section("7. TOTAL TRANSACTION VALUE BY CITY")
ct = df.groupby("City")["Amount"].agg(["count", "sum", "mean"])
ct.columns = ["Transactions", "Total_Value", "Mean_Amount"]
ct["Value_Share_%"] = (ct["Total_Value"] / ct["Total_Value"].sum() * 100).round(2)
ct = ct.sort_values("Total_Value", ascending=False)
say(ct.round(2).to_string())
save(ct.round(2), "07_city_summary")

# --------------------------------------------------------------------------
# 8. Flagged count and percentage
# --------------------------------------------------------------------------
section("8. FLAGGED TRANSACTIONS (supplied label, NOT proof of fraud)")
df["Is_Flagged"] = df["Flagged"].astype(str).str.strip().str.lower().isin(["yes", "y", "1", "true"])
n_flag = int(df["Is_Flagged"].sum())
pct_flag = n_flag / len(df) * 100
say(f"Flagged transactions: {n_flag} of {len(df)} ({pct_flag:.2f}%)")
fl_amt = df.groupby("Is_Flagged")["Amount"].agg(["count", "sum", "mean", "median"])
fl_amt.index = ["Not flagged", "Flagged"]
say(fl_amt.round(2).to_string())
save(fl_amt.round(2), "08_flagged_summary")

# --------------------------------------------------------------------------
# 9. Flagged % by channel
# --------------------------------------------------------------------------
section("9. FLAGGED PERCENTAGE BY CHANNEL")
fch = df.groupby("Channel")["Is_Flagged"].agg(["sum", "count"])
fch.columns = ["Flagged", "Transactions"]
fch["Flagged_%"] = (fch["Flagged"] / fch["Transactions"] * 100).round(2)
fch = fch.sort_values("Flagged_%", ascending=False)
say(fch.to_string())
say("Caution: with small counts per channel, differences in these percentages "
    "may be due to chance.")
save(fch, "09_flagged_by_channel")

# --------------------------------------------------------------------------
# 10. Ten largest transactions
# --------------------------------------------------------------------------
section("10. TEN LARGEST TRANSACTIONS")
cols = ["Transaction_ID", "Transaction_Date", "Account_Type", "Transaction_Type",
        "Channel", "City", "Amount", "Flagged"]
top10 = df.nlargest(10, "Amount")[cols]
say(top10.to_string(index=False))
top10.to_csv(TABLES / "10_top10_transactions.csv", index=False)

# --------------------------------------------------------------------------
# 11. Credit/debit by account type
# --------------------------------------------------------------------------
section("11. CREDIT AND DEBIT TOTALS BY ACCOUNT TYPE")
at = df.groupby(["Account_Type", "Transaction_Type"])["Amount"].agg(["count", "sum", "mean"])
at.columns = ["Transactions", "Total_Amount", "Mean_Amount"]
say(at.round(2).to_string())
save(at.round(2), "11_account_type_credit_debit")
at_pivot = df.pivot_table(index="Account_Type", columns="Transaction_Type",
                          values="Amount", aggfunc="sum")
at_pivot["Net"] = at_pivot["Credit"] - at_pivot["Debit"]
say("\nPivot (total amount):")
say(at_pivot.round(2).to_string())
save(at_pivot.round(2), "11_account_type_pivot")

# --------------------------------------------------------------------------
# 12. Outliers by IQR (np.percentile)
# --------------------------------------------------------------------------
section("12. OUTLIERS USING IQR")
q1, q3 = np.percentile(amt, [25, 75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
say(f"Q1 = {inr(q1)}  Q3 = {inr(q3)}  IQR = {inr(iqr)}")
say(f"Lower fence = {inr(lower)}  Upper fence = {inr(upper)}")
df["Is_Outlier"] = (df["Amount"] < lower) | (df["Amount"] > upper)
n_out = int(df["Is_Outlier"].sum())
say(f"Outliers: {n_out} ({n_out / len(df):.2%} of transactions)")
out_flag_pct = df.loc[df["Is_Outlier"], "Is_Flagged"].mean() * 100 if n_out else 0
non_out_flag_pct = df.loc[~df["Is_Outlier"], "Is_Flagged"].mean() * 100
say(f"Flagged share among outliers: {out_flag_pct:.2f}% vs "
    f"{non_out_flag_pct:.2f}% among non-outliers")
share_value = df.loc[df["Is_Outlier"], "Amount"].sum() / df["Amount"].sum() * 100
say(f"Outliers account for {share_value:.1f}% of total transaction value.")
df[df["Is_Outlier"]].sort_values("Amount", ascending=False)[cols].to_csv(
    TABLES / "12_outliers_iqr.csv", index=False)
pd.DataFrame({"Q1": [q1], "Q3": [q3], "IQR": [iqr], "Lower": [lower], "Upper": [upper],
              "Outliers": [n_out]}).round(2).to_csv(TABLES / "12_iqr_fences.csv", index=False)

# --------------------------------------------------------------------------
# 13. Amount bands
# --------------------------------------------------------------------------
section("13. TRANSACTION AMOUNT BANDS")
df["Amount_Band"] = pd.cut(df["Amount"], bins=BAND_EDGES, labels=BAND_LABELS, right=True)
bands = df.groupby("Amount_Band", observed=False).agg(
    Transactions=("Amount", "count"), Total_Value=("Amount", "sum"),
    Flagged=("Is_Flagged", "sum"))
bands["Count_Share_%"] = (bands["Transactions"] / bands["Transactions"].sum() * 100).round(2)
bands["Value_Share_%"] = (bands["Total_Value"] / bands["Total_Value"].sum() * 100).round(2)
bands["Flagged_%"] = (bands["Flagged"] / bands["Transactions"].replace(0, np.nan) * 100).round(2)
say(bands.round(2).to_string())
save(bands.round(2), "13_amount_bands")

# --------------------------------------------------------------------------
# 14. NumPy Boolean indexing
# --------------------------------------------------------------------------
section(f"14. NUMPY BOOLEAN INDEXING (Amount > {THRESHOLD:,})")
mask = amt > THRESHOLD
above = amt[mask]
say(f"Transactions above threshold: {mask.sum()} ({mask.mean():.2%})")
say(f"Total value above threshold : {inr(above.sum())} "
    f"({above.sum() / amt.sum():.1%} of all value)")
if above.size:
    say(f"Largest: {inr(above.max())}  Smallest in subset: {inr(above.min())}  "
        f"Mean: {inr(above.mean())}")
high = df.loc[mask, cols].sort_values("Amount", ascending=False)
high.to_csv(TABLES / "14_above_threshold.csv", index=False)
say("Breakdown of above-threshold transactions by channel:")
say(high["Channel"].value_counts().to_string())

# --------------------------------------------------------------------------
# Extra analysis: monthly trend and flagged vs amount band
# --------------------------------------------------------------------------
section("EXTRA: MONTHLY TREND")
monthly = df.set_index("Transaction_Date").groupby(
    [pd.Grouper(freq="MS"), "Transaction_Type"])["Amount"].sum().unstack(fill_value=0)
say(monthly.round(0).to_string())
save(monthly.round(2), "extra_monthly_credit_debit")

----------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 15. Cautious observations and recommendations
# --------------------------------------------------------------------------
section("15. OBSERVATIONS AND RECOMMENDATIONS")
top_ch = ch.index[0]
top_city = ct.index[0]
hi_flag_ch = fch.index[0]
lo_flag_ch = fch.index[-1]
recs = [
    (f"Channel mix: {top_ch} has the highest volume ({ch.iloc[0]['Transactions']:.0f} "
     f"transactions, {ch.iloc[0]['Volume_Share_%']:.1f}% of all).",
     "Review capacity and monitoring coverage for the busiest channels first, and "
     "track whether the mix shifts over time before drawing conclusions."),
    (f"City value: {top_city} has the highest total value "
     f"({ct.iloc[0]['Value_Share_%']:.1f}% of value). Differences between cities may "
     "reflect customer mix and a few large transactions rather than behaviour.",
     "Compare cities on median and per-account values as well as totals, so a handful "
     "of large transactions does not drive the picture."),
    (f"Flagged label: {n_flag} records ({pct_flag:.2f}%) carry the supplied flag. "
     f"{hi_flag_ch} shows the highest flagged share ({fch.iloc[0]['Flagged_%']:.1f}%) "
     f"and {lo_flag_ch} the lowest ({fch.iloc[-1]['Flagged_%']:.1f}%). Flags are a "
     "supplied label, not proof of fraud, and counts per channel are small.",
     "Treat flags as a prioritisation aid for manual review only. Document how the flag "
     "is assigned and review it with the owning team before any action."),
    (f"Large values: the IQR rule marks {n_out} transactions ({n_out / len(df):.1%}) "
     f"as outliers, holding {share_value:.1f}% of total value. Large amounts are "
     "unusual in this data but not wrong by themselves.",
     "Review the outlier list for data-entry or unit errors and check whether large "
     "amounts have a legitimate explanation (e.g. business accounts) before escalation."),
    (f"Data quality: {n_extra} repeated Transaction_IDs and {n_missing} missing amounts "
     "were found. Median imputation keeps counts complete but adds estimated values.",
     "Add uniqueness and not-null checks at data load, and report imputed rows "
     "separately so totals can be read with that uncertainty in mind."),
]
for i, (obs, rec) in enumerate(recs, 1):
    say(f"\n{i}. Observation: {obs}\n   Recommendation: {rec}")
say("\nReminder: the dataset is synthetic. These notes describe transaction "
    "behaviour only and must not be used for fraud claims.")

with open(OUT / "recommendations.md", "w", encoding="utf-8") as f:
    f.write("# Cautious Operational Observations and Recommendations\n\n")
    f.write("_Synthetic dataset. Flagged is a supplied label, not proof of fraud._\n\n")
    for i, (obs, rec) in enumerate(recs, 1):
        f.write(f"{i}. **Observation:** {obs}\n   **Recommendation:** {rec}\n\n")

(OUT / "analysis_report.txt").write_text("\n".join(report), encoding="utf-8")
print(f"\nDone. Outputs written to {OUT}")
