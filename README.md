# Project 10: Banking Transaction Pattern Analysis

Industry: Banking / FinTech

Exploratory analysis of transaction amounts, channels, transaction types and
flagged records.

> **Important:** the dataset is synthetic. The `Flagged` column is a supplied
> label, not proof of fraud. Nothing in this project should be used to make
> fraud claims about any person or account. Amounts are treated as INR for
> readability only.

## Project structure

```
bank_txn_analysis/
├── data/10_bank_transactions.csv     # synthetic input (1,208 rows)
├── src/analysis.py                   # full analysis, tasks 1-15
├── src/generate_data.py              # script that created the synthetic data
├── outputs/
│   ├── analysis_report.txt           # full printed report
│   ├── recommendations.md            # task 15
│   ├── cleaned_transactions.csv      # after de-duplication and imputation
│   ├── tables/                       # one CSV per task
│   └── charts.html                   # charts (plain HTML/CSS, open in a browser)
├── docs/SRS.md                       # software requirements specification
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python src/analysis.py
```

Outputs are written to `outputs/`. To rebuild the dataset: `python src/generate_data.py`.

## Task coverage

| # | Task | Where |
|---|------|-------|
| 1 | Load and inspect | report section 1 |
| 2 | Missing amounts | section 2 (median by Account_Type + Transaction_Type; rows marked `Amount_Imputed`) |
| 3 | Duplicate Transaction_ID | section 3, `tables/03_duplicate_transactions.csv` |
| 4 | Total credits and debits | section 4 |
| 5 | Mean, median, std | section 5 |
| 6 | Volume by channel | section 6, `tables/06_channel_summary.csv` |
| 7 | Value by city | section 7, `tables/07_city_summary.csv` |
| 8 | Flagged count and % | section 8 |
| 9 | Flagged % by channel | section 9 |
| 10 | Ten largest transactions | section 10 |
| 11 | Credit/debit by account type | section 11 (`groupby(['Account_Type','Transaction_Type'])`) |
| 12 | IQR outliers | section 12 (`np.percentile`) |
| 13 | Amount bands | section 13 |
| 14 | NumPy Boolean indexing | section 14 (threshold 100,000) |
| 15 | Recommendations | section 15, `recommendations.md` |





## Method notes

- **Order of cleaning:** duplicates are removed first (keep first), then missing
  amounts are imputed, so no value is estimated for a row that is dropped.
- **Imputation:** group median is used because amounts are strongly right-skewed.
  Imputed values are estimates and add uncertainty to totals.
- **Outliers:** IQR rule, fences at Q1 - 1.5 IQR and Q3 + 1.5 IQR. Because the
  data is skewed, many legitimate large transactions are marked as outliers.
- **Bands:** upper edge inclusive, so band `>100K` matches the Boolean-indexing
  threshold in task 14 exactly.
- **Small counts:** flagged percentages by channel rest on small counts and may
  differ by chance.
