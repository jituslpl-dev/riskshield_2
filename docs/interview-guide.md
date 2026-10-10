# Interview walkthrough

1. Explain the problem: calibrated adverse-event probability is distinct from the action chosen under asymmetric costs.
2. Run tests and show the point-in-time velocity check. Explain why an account's current/simultaneous events cannot enter its history.
3. Walk through the five data partitions. Explain why calibration, model selection, threshold selection and final testing have different responsibilities.
4. Compare PR-AUC with the event-rate baseline; explain why accuracy and ROC-AUC alone can be misleading with rare fraud.
5. Show the real benchmark result and its limitations. The ULB test has 22 positives; UCI has no application chronology.
6. Compare expected costs with imperfect review to the legacy threshold strategy. Describe how queue capacity would change the optimization.
7. Register two model versions, promote the newer version, then roll back. Explain corruption detection versus artifact trust.
8. Retry a prediction with the same idempotency key and show one audit entry; show a 409 for altered input.
9. Explain the repository Protocol, service composition, classifier factory and immutable policy object with their actual source files.
10. Show SHAP additivity on the raw probability scale and explain why calibration changes the final score.
11. Discuss PSI versus supervised performance monitoring, label maturity, fairness, and costs of false declines.
12. State scaling boundaries: Python lock serializes prediction auditing; SQLite is not the distributed design. Describe an atomic PostgreSQL idempotency transaction and object-store registry as future replacements.

Useful extensions to discuss, without claiming implementation: rolling-window evaluation on richer historical data; transaction-specific expected loss; review queue allocation; delayed-label monitoring; signed artifact promotion; external feature stores; counterfactual explanations for interpretable feature schemas.
