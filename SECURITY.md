# Security Policy

CARE-RAG includes code for reproducing and evaluating corpus-poisoning attacks
against RAG systems. The attack components are included to support defensive
research and benchmark reproducibility.

## Responsible Disclosure

If you find a vulnerability in this repository or identify a way that released
artifacts could enable unintended harm, please report it privately to the
maintainers before public disclosure.

Do not include live third-party targets, private data, API keys, or operational
attack instructions in public issues.

## Scope

In scope:

- Bugs that leak credentials or private local paths.
- Unsafe default behavior that writes harmful artifacts outside the workspace.
- Reproducible failures in the defense or evaluation code that materially affect
  reported security conclusions.

Out of scope:

- Attacks against third-party services using this code.
- Requests to generate or publish poisoned content against real systems.
- Model behavior issues unrelated to this repository's code.
