# Contributing

Thanks for your interest in improving **Skill Is Not Document**. This guide explains how to file
issues, propose features and designs, and submit pull requests.

## Ways to contribute

- Report a bug or unexpected result
- Request a feature or improvement
- Improve documentation or examples
- Submit code (bug fix, new feature, optimization)
- Contribute data or evaluation cases

## Filing an issue

Open a GitHub issue and include:

- **Bug**: what you ran (command/code), what you expected, what happened, and the environment
  (OS, Python, `torch` / `sentence-transformers` versions, GPU). A minimal reproducible example helps.
- **Feature request**: the problem you are trying to solve, why the current behavior is insufficient,
  and a sketch of the proposed behavior.
- **Question**: please search existing issues first.

Use clear, descriptive titles. One issue per topic.

## Proposing a design

For non-trivial changes (new modules, API changes, training/eval protocol changes), open a
**design issue** before coding:

1. Describe the motivation and scope.
2. Sketch the approach (interfaces, data flow, alternatives considered).
3. Note compatibility / reproducibility impact — especially anything that could change the
   train/inference prompts or the reported metrics.
4. Wait for maintainer feedback before a large PR.

This avoids wasted effort and keeps the published numbers reproducible.

## Submitting a pull request

1. Fork the repo and create a topic branch (`fix/...`, `feat/...`, `docs/...`).
2. Keep the PR focused; one logical change per PR.
3. Make sure the smoke test passes:
   ```bash
   python3 infer.py --query "I need to compose music" --skills example_skills.jsonl --recall_n 6 --top_k 3
   ```
4. If you touched retrieval/eval logic, confirm `reproduce.py` still matches the paper numbers and
   report the before/after in the PR description.
5. Match the existing code style (plain, dependency-light, no unused code).
6. Update docs (`README.md`) when behavior or flags change.
7. Write a clear PR description: what, why, and how it was tested. Link the related issue.

## Code style

- Python 3.10+, standard library + `torch` / `sentence-transformers` / `numpy` only.
- Keep scripts runnable as `python3 <file>.py` with sensible defaults.
- Do not change the hard-coded train/inference prompt strings unless the design issue explicitly
  covers retraining; they are part of train/inference consistency.

## License of contributions

By contributing, you agree that your contributions are licensed under the Apache License 2.0 (see
[LICENSE](LICENSE)).
