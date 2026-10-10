# Architecture decisions

## ADR 001: domain logic uses composition

Decision: keep `CostPolicy` independent of FastAPI, databases and sklearn. Inject it into `PredictionService`. Classifier creation is centralized in a factory; persistence is represented by a Protocol and concrete filesystem adapter.

Reason: tests can exercise policies and failure handling without a running server or real datasets. A future object-store adapter can implement the same repository contract. Tradeoff: experiment orchestration still uses the original numerical pipeline rather than a fully abstract training framework.

## ADR 002: model artifacts have immutable version directories

Decision: registration fails when a version exists; active pointers are atomically replaced after checksum/schema validation. Promotion is an explicit CLI operation rather than automatic training behavior.

Reason: deployment and rollback are separate from model fitting. Tradeoff: local files do not provide distributed release coordination, cryptographic signing, or automatic approval gates. Cached versions persist in memory until process restart.

## ADR 003: SQLite is a single-process audit adapter

Decision: store decisions and input fingerprints, avoid raw feature rows, and serialize prediction/idempotency transactions with a process lock.

Reason: local durability and concurrent-retry behavior are executable without external services. Tradeoff: this also serializes scoring through that service and limits throughput. Multiple processes are not protected by the Python lock; move to a transactional PostgreSQL adapter with atomic uniqueness handling for multi-worker use. Audit retention and access controls remain operational responsibilities.

## ADR 004: retain two policy strategies

Decision: preserve original held-out threshold tuning and add expected-cost comparison with imperfect review. Costs/review accuracy are configuration, not fitted to final test outcomes.

Reason: selection by cost is easier to discuss when review assumptions are explicit. Tradeoff: neither strategy models queue capacity, delayed fraud labels, or transaction-specific recovery rates. Those are extensions, not claimed capabilities.

## ADR 005: schema-driven endpoints serve benchmark models

Decision: every registered model stores feature names and exposes them through metadata. The generalized endpoint requires exact matches. Demo aliases retain their original domain schemas.

Reason: PCA fraud records and UCI repayment records cannot honestly be treated as the same schema as synthetic transactions or loans. Tradeoff: general endpoints enforce numerical structure, not every business constraint; a production dataset needs a dedicated validated schema.

## ADR 006: no test-set performance selection

Decision: preserve originally selected models, add bootstrap/PSI/reliability diagnostics without retuning using test labels, and retain unfavorable baseline comparisons.

Reason: transparent measurement is more useful than optimized reporting. Tradeoff: single splits and sparse fraud positives have large uncertainty, and row bootstrap ignores serial dependence. Longer datasets and rolling evaluation are needed before operational use.
