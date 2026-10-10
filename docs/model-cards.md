# Model cards

## Fraud demo

Purpose: exercise the pipeline and API using synthetic accounts and transactions. Inputs include historical account velocity. Limitation: generator-defined labels and small positive counts do not establish real-world discrimination or calibration.

## Credit demo

Purpose: synthetic application risk and separate API schema. Inputs include debt-to-income and credit history. Limitation: original policy sent every test application to review under the perfect-review cost assumption.

## ULB/Worldline fraud benchmark

Purpose: historical benchmark evaluation with chronological splits. Features: Amount and 28 publisher-provided PCA components. Selected model: Logistic Regression with isotonic calibration. Final test: 28,376 rows, 22 frauds, average precision 0.62050 and Brier 0.000362. No account IDs; no real velocity validation; only two days of 2013 data. PCA preprocessing provenance cannot be fully audited.

## UCI credit default benchmark

Purpose: default-next-month classification for existing credit-card accounts. Features: limit, repayment statuses, bill amounts and payment amounts; demographic fields excluded. Selected model: CatBoost with Platt scaling. Final test: 2,997 rows, 663 defaults, average precision 0.57665 and Brier 0.13147. Five stratified partitions, no temporal-generalization claim. Original policy cost exceeded the review-all baseline. Demographic exclusion is not a fairness assessment.

All model binaries are locally generated and ignored by Git. Reports record exact sources, schemas, hashes and limitations. Use only trusted artifacts under matching package versions. None of these models is approved for live lending or fraud decisions.
