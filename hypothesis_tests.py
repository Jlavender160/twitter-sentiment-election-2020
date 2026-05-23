"""
H1: Sentiment distribution differs before vs after the election
H2: Trump-related tweets are more negative than Biden-related tweets

Method: Chi-square test of independence + Cramér's V effect size
A p-value below 0.05 means we reject the null hypothesis (H₀)
and conclude the difference is statistically significant.

Inputs:  breakdowns/sentiment_by_period.csv
 breakdowns/sentiment_by_candidate.csv
Outputs: breakdowns/hypothesis_test_results.txt
"""

import io, os, sys
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR     = "/Users/james/Downloads/archive1"
BREAKDOWNS   = os.path.join(BASE_DIR, "breakdowns")
RESULTS_FILE = os.path.join(BREAKDOWNS, "hypothesis_test_results.txt")
ALPHA        = 0.05  # significance threshold — p < 0.05 means we reject H₀


def run_test(title, csv_file, index_col, row_keys, row_labels):
    """Run a chi-square test on a 2×3 sentiment contingency table."""
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

    df    = pd.read_csv(os.path.join(BREAKDOWNS, csv_file)).set_index(index_col)
    table = df.loc[row_keys, ["negative", "neutral", "positive"]].values

    # Print contingency table
    print(f"\n  {'':10} {'Negative':>10} {'Neutral':>10} {'Positive':>10} {'Total':>10}")
    for label, row in zip(row_labels, table):
        pct = f"  ({100*row[0]/row.sum():.1f}% neg)" if len(row_labels) == 2 else ""
        print(f"  {label:10} {row[0]:>10,} {row[1]:>10,} {row[2]:>10,} {row.sum():>10,}{pct}")

    # Chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(table)
    decision  = "REJECT H₀ (significant)" if p < ALPHA else "FAIL TO REJECT H₀"
    cramers_v = np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
    magnitude = "small" if cramers_v < 0.1 else "medium" if cramers_v < 0.3 else "large"

    print(f"\n  Chi-square statistic : {chi2:.4f}")
    print(f"  Degrees of freedom   : {dof}")
    print(f"  p-value              : {p:.6f}")
    print(f"  Min expected count   : {expected.min():.2f}  {'✓ Cochran OK' if expected.min() >= 5 else '⚠ Cochran warning'}")
    print(f"  α = {ALPHA}  →  {decision}")
    print(f"  Cramér's V           : {cramers_v:.4f}  ({magnitude})")

    return {"statistic": chi2, "p": p, "dof": dof, "decision": decision, "effect": cramers_v}


def main():
    print("=" * 60)
    print("  Dissertation Hypothesis Tests — US Election 2020")
    print(f"  Significance level: α = {ALPHA}")
    print("=" * 60)

    # Capture output to both stdout and file simultaneously
    buf = io.StringIO()
    _real_stdout = sys.stdout
    class Tee:
        def write(self, msg): _real_stdout.write(msg); buf.write(msg)
        def flush(self): _real_stdout.flush()
    sys.stdout = Tee()

    # TEST H1: Did sentiment shift significantly before vs after the election?
    h1 = run_test("H1: Sentiment Shift — Before vs After Election",
                  "sentiment_by_period.csv", "period",
                  ["before", "after"], ["Before", "After"])

    # TEST H2: Are Trump tweets significantly more negative than Biden tweets?
    h2 = run_test("H2: Trump vs Biden Sentiment Distribution",
                  "sentiment_by_candidate.csv", "candidate",
                  ["trump", "biden"], ["Trump", "Biden"])

    # Summary table
    print(f"\n{'='*60}\n  HYPOTHESIS TEST SUMMARY\n{'='*60}")
    print(f"\n  {'':4} {'Hypothesis':50} {'p-value':>10}  Decision")
    print(f"  {'-'*4} {'-'*50} {'-'*10}  {'-'*30}")
    for hyp, desc, result in [
        ("H1", "Sentiment shifts before → after election", h1),
        ("H2", "Trump tweets more negative than Biden tweets", h2),
    ]:
        print(f"  {hyp:4} {desc:50} {result['p']:>10.6f}  {result['decision']}")

    sys.stdout = sys.__stdout__
    os.makedirs(BREAKDOWNS, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        f.write(buf.getvalue())
    print(f"\n  Results saved to {RESULTS_FILE}")
    print(f"\n{'='*60}\n  DONE\n{'='*60}")


if __name__ == "__main__":
    main()
