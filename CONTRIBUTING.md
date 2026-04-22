# Contributing to SynthEdge

Thanks for your interest in contributing. SynthEdge is a research-grade tool — we value correctness, clarity, and honest benchmarks above all.

## Ways to contribute

**Bug reports** — Open an issue with a minimal reproducible example. Include your dataset shape, Python version, and the full traceback.

**New datasets for the benchmark** — If you run SynthEdge on a real-world imbalanced dataset and get results worth sharing, open a PR adding them to `benchmarks/`. Datasets from healthcare, fraud detection, and anomaly detection are especially welcome.

**Synthesis backends** — The `synthesizer.py` module is designed to be modular. If you want to add a new synthesis method (diffusion model, VAE, etc.) as a backend option, open a discussion first.

**Gap report UI** — The HTML gap report is not yet built. If you want to take this on, see the issue tracker for the spec.

## Development setup

```bash
git clone https://github.com/Juzt-nik/SynthEdge.git
cd SynthEdge
pip install -e ".[dev]"
pip install ctgan imbalanced-learn xgboost hdbscan
```

## Running tests

```bash
pytest tests/ -v
```

All 35 tests should pass. If you add a feature, add a test for it.

## Code style

- Black formatting (`black synthedge/`)
- No f-strings with complex expressions inside notebook-destined code
- All public functions need a docstring
- Keep synthesis, scanning, and quality as separate modules — don't merge concerns

## Benchmark integrity

SynthEdge's credibility depends on honest benchmarks. If you find a case where SynthEdge performs worse than SMOTE, report it — don't hide it. The severity classifier exists precisely to warn users when SynthEdge won't help.

## Pull request checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] New functionality has tests
- [ ] Docstrings updated
- [ ] README updated if user-facing API changed

## Questions?

Open an issue tagged `question`. Response within a few days.
