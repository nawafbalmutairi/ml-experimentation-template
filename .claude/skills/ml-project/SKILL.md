---
name: ml-project
description: Use when working on a dataset or ML problem - profiling data, choosing a problem type, split strategy, features or model, evaluating results, or deciding what to keep from a finished project.
---

# ML project workflow

## Overview

Work one stage at a time. Each stage ends with a recommendation and a decision that belongs to the
user. Claude supplies analysis and options; the user chooses.

**Surfacing a decision is never optional. Waiting for it depends on whether the user can answer.**

- If the user is reachable, stop after each stage and wait.
- If the user has said they are away or cannot answer, keep going rather than stalling: state the
  assumption you are proceeding on, name the alternative you rejected, and carry both into the
  report so the decision is still theirs to revisit.

A choice made silently is the failure, not a choice made without waiting.

## Every stage delivers the same three parts

1. **What I found** - evidence and numbers, not impressions.
2. **Options** - the real alternatives, with their trade-offs.
3. **My recommendation** - and the reasoning behind it.

Then stop and wait. The user's answer opens the next stage.

## Stages

| # | Stage | Deliver |
| --- | --- | --- |
| 0 | Intake | Objective in business terms; **prediction point**; constraints |
| 1 | Profile | Facts / Observations / Risks / Unknowns |
| 2 | Framing | Classification, regression or clustering, and the alternatives |
| 3 | Split | Random, stratified, temporal, hold-out validation or CV, with leakage implications |
| 4 | Features | Keep / Drop / Transform / Create, each checked against the prediction point |
| 5 | Model | Baseline first, then candidates and their trade-offs |
| 6 | Train + evaluate | Metrics, baseline comparison, error analysis, business impact |
| 7 | Report | Written to `reports/<project>.md` |
| 8 | Backport | What is generic enough to belong in the framework |

## Stage detail

**0 - Prediction point.** State the moment the prediction is made and what is known then. Every
feature in stage 4 is checked against it. A column recorded during or after the outcome fails, no
matter how predictive.

**1 - Profile.** Cover shape and dtypes; missingness *including sentinels* (`unknown`, `-1`, `9999`,
empty string); duplicates, and whether duplicates disagree on the target; target distribution and
balance; categorical cardinality, flagging anything that would explode under one-hot; numeric ranges
and whether extremes are real; whether rows are in time order and whether the target drifts along
it; leakage suspects; and what is missing that the objective needs.

**3 - Split.** Tune against validation data and open the test set once, at the end. If time
ordering exists, say what a random split would cost before recommending one.

**6 - Evaluate.** Compare against the baseline the *decision* faces - calling at random, or
predicting the mean - not only a majority-class number. Report weak results as weak. State when a
number is not trustworthy and why.

**7 - Report.** A file, not a chat message. Decisions, evidence, results, limitations.

## Framework and project code are separate

Project configs, scripts and reports are project files. If you change anything under
`ml_template/`, `tests/` or `scripts/verify.py`, append one line to `BACKPORT.md` saying what and
why, at the time you change it. Stage 8 reads that file instead of reconstructing history.

## Iteration

Returning to features or model after error analysis is normal, not failure. Say that you are going
back, and record how many passes it took - the report states this.

## Common mistakes

- Running stages together while the user is available, which takes decisions away from them.
- Stalling for an answer from someone who has said they are unreachable.
- Recommending without giving the alternatives that were rejected.
- Profiling with `describe()` and calling it a profile.
- Leaving a capability gap silent: if the framework cannot do what the data needs, say so.
