# ML project workflow — design

Date: 2026-08-18
Status: proposed, awaiting review

## Problem

This repository works well for one dataset at a time, but nothing defines how the *next* dataset
starts. Two failure modes follow from that gap:

1. **Project drift.** Without a rule, project #9 gets copied from project #8, which was copied from
   #7. Each copy carries the previous problem's configs, reports and half-finished ideas.
2. **Process amnesia.** The care taken on a dataset — profiling before modelling, catching the
   `duration` leak, noticing the temporal drift — lives only in a conversation. A fresh session six
   months from now has none of it, and will happily jump straight to a model.

The code is in good shape. The repeatability around it is not.

## Decisions

### D1 — The repository is a reference; every project copies it

`ml-experimentation-template` is **#0**: the canonical framework. Each dataset gets its own copy of
#0. A project is **never** copied from another project. Project #9 and project #2 begin from the
same base; the only differences between them are the ones the data demanded.

### D2 — #0 is a skeleton, not a toolkit

#0 contains the machinery, not a menu of every technique. No speculative clustering support, no
model zoo sitting unused. Capabilities arrive when a real dataset needs one.

### D3 — The workflow lives in a skill

`.claude/skills/ml-project/SKILL.md`, so it is invoked automatically, survives fresh sessions, and
travels with every copy.

### D4 — Claude runs the process; the user makes the decisions

The skill governs **when Claude must stop and ask**. It does not choose the problem type, the split,
the model, or the threshold. Every fork ends with the user's decision.

### D5 — Backporting is the only path between projects

Nothing flows sideways. A capability built in project #3 reaches project #4 only by being pushed
back into #0 first. This makes step 8 load-bearing rather than optional.

## Skeleton contents

| Kept in #0 | Reason |
| --- | --- |
| `ml_template/` — config, data_processor, data_validator, feature_engineer, model, train, evaluate, predict | The engine |
| `config/config.yaml`, `data/raw/sample.csv` | A fresh copy runs immediately; CI has something to exercise |
| `tests/`, `scripts/verify.py`, `.github/workflows/ci.yml` | The quality floor every copy inherits |
| `scripts/lift_analysis.py` | Generic decile lift for any binary ranking problem |
| `docs/superpowers/specs/` | Design decisions, including this one |

No project-specific config, report or dataset is ever committed to #0.

## Copy procedure

No tooling. Copying a directory is not a problem that needs a script; one is justified only if this
becomes painful in practice.

1. Confirm #0 is green: `python scripts/verify.py` passes.
2. Copy #0 to a new directory named for the project.
3. Delete `.git`, run `git init`.
4. Drop the dataset into `data/raw/`.
5. Confirm the copy is green before touching anything.

Step 1 is a precondition, not a formality: a copy inherits whatever state #0 is in, so #0 is never
left broken.

## The workflow

Eight steps. The stop column marks where Claude presents options and waits for a decision.

| Step | Claude produces | Stop |
| --- | --- | --- |
| 0. Intake | The goal restated in business terms: what decision does this model serve, and what does acting on it cost? | confirm |
| 1. Profile | The findings report (below) | findings and gaps |
| 2. Framing | Classification, regression or clustering, with reasoning from the profile | **user decides** |
| 3. Split | Random, stratified, chronological, hold-out validation or cross-validation, recommended from the profile | **user decides** |
| 4. Features | Columns in, columns out and why, sentinel and missing-value handling | **user decides** |
| 5. Model | A baseline first, then a candidate family and parameters | **user decides** |
| 6. Train and evaluate | Metrics, the baseline beside them, and a business-relevant framing (lift, or error in real units) | good enough? |
| 7. Report | `reports/<project>.md`: decisions, evidence, limitations | review |
| 8. Backport | What is generic, and a proposal to move it into #0 | confirm |

### Step 1 profile — required contents

The report is not a `df.describe()` dump. It must cover:

- Shape, dtypes, memory
- Missingness per column, **including sentinel values** (`unknown`, `-1`, `9999`, empty string)
- Duplicate rows, and whether duplicates disagree on the target
- Target: distribution and class balance, or range and skew for regression
- Categorical cardinality, flagging any column that would explode under one-hot encoding
- Numeric ranges, and whether extremes are errors or real
- **Time**: any date column, whether rows are chronologically ordered, and whether the target drifts
  across that order
- **Leakage suspects**: columns knowable only after the outcome, identifiers, and anything
  suspiciously predictive
- What is missing that the stated goal needs

### Step 6 — baseline is mandatory

Every model is reported next to a trivial baseline: majority class for classification, mean or
median for regression. A model that does not beat it is reported as not beating it.

## What the skill binds Claude to

1. Profile before proposing a model. No reaching for the previous project's algorithm.
2. Stop at every checkpoint with real options, a recommendation, and the reasoning.
3. Always report a baseline comparison.
4. Check leakage explicitly at step 1 and again at step 4.
5. Name judgement calls out loud, including small ones.
6. Report weak results as weak, and say when a number is not trustworthy.
7. State when the skeleton lacks a capability the data needs, rather than working around it silently.

## Out of scope, deliberately

Clustering, cross-validation, three-way train/validation/test splits, additional model families,
date-part extraction, high-cardinality encoding, hyperparameter search, calibration, feature
importance output.

Each is built when a dataset needs it, in that project, then backported if generic. Date handling
and a cardinality guard are the most likely first backports, since most real tables contain a date
column and at least one high-cardinality identifier.

## Verification

The skill is prose and has no unit tests. Acceptance is behavioural:

1. #0 stays green: `scripts/verify.py` passes after the skill is added.
2. A fresh copy of #0 trains on `sample.csv` unchanged.
3. A dry run of the skill against `sample.csv` stops at every checkpoint and produces a profile
   report before any model is proposed.

## Risks and trade-offs

| Risk | Assessment |
| --- | --- |
| Copies drift apart | Accepted; inherent to the model chosen. D5 is the mitigation and depends on discipline |
| The skill guarantees the checkpoint, not the judgement | It makes Claude stop and ask; it does not make the recommendation correct |
| #0 has no worked example any more | The reasoning lives in project repos instead. Accepted to keep #0 clean |
| Skeleton gaps surface mid-project | Expected. Binding rule 7 requires naming the gap rather than hacking around it |
| Checkpoint fatigue | Eight stops may feel heavy on a simple dataset. If so, merge steps 4 and 5 rather than skipping them silently |

## Open questions

1. **Where do project copies live?** Sibling directories, or their own GitHub repositories? This
   affects nothing in the design but should be settled before project #1.
2. **Do project repos get remotes?** #0 is public; a project containing client data may not want to
   be.
