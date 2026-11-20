# Cautious Operational Observations and Recommendations

_Synthetic dataset. Flagged is a supplied label, not proof of fraud._

1. **Observation:** Channel mix: UPI has the highest volume (305 transactions, 25.4% of all).
   **Recommendation:** Review capacity and monitoring coverage for the busiest channels first, and track whether the mix shifts over time before drawing conclusions.

2. **Observation:** City value: Mumbai has the highest total value (21.9% of value). Differences between cities may reflect customer mix and a few large transactions rather than behaviour.
   **Recommendation:** Compare cities on median and per-account values as well as totals, so a handful of large transactions does not drive the picture.

3. **Observation:** Flagged label: 43 records (3.58%) carry the supplied flag. Net Banking shows the highest flagged share (9.2%) and UPI the lowest (1.3%). Flags are a supplied label, not proof of fraud, and counts per channel are small.
   **Recommendation:** Treat flags as a prioritisation aid for manual review only. Document how the flag is assigned and review it with the owning team before any action.

4. **Observation:** Large values: the IQR rule marks 94 transactions (7.8%) as outliers, holding 70.3% of total value. Large amounts are unusual in this data but not wrong by themselves.
   **Recommendation:** Review the outlier list for data-entry or unit errors and check whether large amounts have a legitimate explanation (e.g. business accounts) before escalation.

5. **Observation:** Data quality: 8 repeated Transaction_IDs and 24 missing amounts were found. Median imputation keeps counts complete but adds estimated values.
   **Recommendation:** Add uniqueness and not-null checks at data load, and report imputed rows separately so totals can be read with that uncertainty in mind.

