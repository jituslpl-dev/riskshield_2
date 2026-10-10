# Operational runbook

## Service not ready

`/live` healthy and `/ready` 503 usually means models have not been registered/promoted. Run `python -m riskshield.cli list`, inspect active pointers, and promote a verified version. No model is fabricated by the API.

## Rollback

Run `python -m riskshield.cli promote --name fraud_demo --version PREVIOUS_VERSION`. New loads observe the pointer. A retry of an existing idempotency key against a different model version returns 409; issue a new key only when a genuinely new prediction is intended.

## Artifact corruption

Stop promotion and regenerate from a trusted training run. Never bypass the checksum or load an unknown joblib file. Preserve incident evidence and compare the manifest/source build. The manifest itself must be protected from unauthorized writes.

## Drift

Use `population_stability_index(reference_scores,current_scores)` with at least 100 probabilities in each window. Investigate alerts against seasonality, traffic mix and data-quality changes. PSI alone is not a retraining trigger. Confirm performance changes using matured labels and independent validation windows.

## Audit and metrics

`runtime/audit.sqlite` holds fingerprints and decisions. Treat it as sensitive, configure retention/backups and restrict filesystem access. Metrics are process-local counters; multiple workers require aggregation. Do not use the sample latency report as a deployment SLA.

## Deployment boundary

Docker/Compose are provided as a non-root scaffold. TLS, reverse-proxy limits, distributed idempotency, managed secrets, role-based promotion, fairness validation and operational loss estimates are not supplied by a container. Validate them separately on the target infrastructure.
