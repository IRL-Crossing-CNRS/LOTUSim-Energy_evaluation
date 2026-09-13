# Contributing

This repository records an evaluation, so contributions are held to one
standard above all: a change must keep every reported number traceable to the
runs that produced it.

## Commits

- Sign off every commit: `git commit -s`.
- Write plain, factual messages — what changed and why, in bullet points.
- One logical change per commit.

## Changing analysis code

A change to how a metric is computed changes what the stored results mean.
When you make one:

1. Say in the commit message which stored numbers it invalidates.
2. Regenerate them (`aerial-navigation/scripts/collect_results.py`, or the
   relevant `wake-models/benchmark/` script) in the same commit or the next one.
3. Never edit a stored result by hand.

## Adding data

Image and binary data belongs in Git LFS — see `.gitattributes`. Keep model
weights and training run directories out of the repository entirely; they are
regenerated from the committed configuration.

## The simulator repositories

Changes to LOTUSim, LOTUSim-generic-scenario or LOTUSim-Energy belong in those
repositories, not here. If a study starts depending on a capability one of them
does not have yet, land it there first and record the requirement in
`docs/upstream-requirements.md`; do not carry a patch in this repository.
