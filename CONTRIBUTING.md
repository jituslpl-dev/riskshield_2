# Contributing

Install locked runtime and development dependencies, then install the package in editable mode. Run `ruff check src tests scripts` and `python -m pytest -q` before proposing a change. Keep domain policies independent of HTTP and infrastructure code.

Include a meaningful regression test for changed decision behavior, schema handling, concurrency or artifact lifecycle. Do not upload datasets, trained joblib binaries, secrets or audit databases. Document new data sources, split protocols and benchmark limitations in the README and model cards. Never select a model using final test metrics.

Container or deployment changes should pass the Linux Docker smoke job. Numerical changes should reproduce both synthetic training commands; real-data re-evaluation remains an explicit task because datasets are large.
