# ML project workflow — design

Date: 2026-08-18
Status: implemented as `.claude/skills/ml-project/SKILL.md`. Two rules verified against baselines,
the checkpoint rule reworded after testing, one criterion still untested.

## Problem

This repository works well for one dataset at a time, but nothing defined how the *next* dataset
starts. Two failure modes follow:

1. **Project drift.** Without a rule, project #9 gets copied from #8, which was copied from #7. Each
   copy carries the previous problem's configs, reports and half-finished ideas.
2. **Process amnesia.** The care taken on a dataset — profiling before modelling, catching a leaking
   column, noticing temporal drift — lives only in a conversation. A fresh session months later has
   none of it.

## Decisions

### D1 — The repository is a reference; every project copies it

`ml-experimentation-template` is **#0**. Each dataset gets its own copy of #0. A project is **never**
copied from another project, so project #9 and project #2 begin from the same base and differ only
where the data demanded it.

### D2 — #0 is a skeleton, not a toolkit

The machinery, not a menu of techniques. Capabilities arrive when a real dataset needs one.

### D3 — The workflow lives in a skill

`.claude/skills/ml-project/SKILL.md`, so it is invoked automatically, survives fresh sessions, and
travels with every copy.

### D4 — Claude runs the process; the user makes the decisions

Claude supplies findings, options and a recommendation. The user chooses.

### D5 — Backporting is the only path between projects

Nothing flows sideways. A capability built in project #3 reaches project #4 only by going back into
#0 first.

## Skeleton contents

| Kept in #0 | Reason |
| --- | --- |
| `ml_template/` | The engine |
| `config/config.yaml`, `data/raw/sample.csv` | A fresh copy runs immediately; CI has something to exercise |
| `tests/`, `scripts/verify.py`, `.github/workflows/ci.yml` | The quality floor every copy inherits |
| `scripts/lift_analysis.py` | Generic decile lift for any binary ranking problem |
| `.claude/skills/ml-project/` | The workflow |
| `docs/superpowers/specs/` | Design decisions, including this one |

No project-specific config, report or dataset is committed to #0.

## Copy procedure

1. Confirm #0 is green: `python scripts/verify.py` passes.
2. Copy #0 to a directory named for the project.
3. Delete `.git`, run `git init`.
4. Drop the dataset into `data/raw/`.
5. Confirm the copy is green before touching anything.

Step 1 is a precondition: a copy inherits whatever state #0 is in.

## The workflow

Every stage delivers the same three parts — **what I found**, **the options**, **my recommendation
and why** — then the decision belongs to the user.

| Step | Contents | Decision |
| --- | --- | --- |
| 0. Intake | Business objective · **prediction point** · constraints | confirm |
| 1. Profile | Facts · Observations · Risks · Unknowns | review |
| 2. Framing | Classification / regression / clustering · alternatives + reasoning | **user decides** |
| 3. Split | Random / stratified / temporal / validation / CV · leakage implications | **user decides** |
| 4. Features | Keep / Drop / Transform / Create · leakage check against the prediction point | **user decides** |
| 5. Model | Baseline · candidates · trade-offs | **user decides** |
| 6. Train + evaluate | Metrics · baseline comparison · error analysis · business impact | review |
| 7. Report | Decisions · evidence · results · limitations, written to a file | review |
| 8. Backport | Generic capability? | confirm |

### Stopping is conditional; surfacing is not

If the user is reachable, stop after each stage and wait. If they have said they are away, proceed
under a stated assumption rather than stalling, name the rejected alternative, and carry both into
the report. **A choice made silently is the failure; a choice made without waiting is not.**

### Iteration

Error analysis at step 6 routinely sends work back to features or model. That is expected. Say that
you are going back, and record how many passes it took.

### Test-set discipline

Tune against validation data. Open the test set once, at the end. Iterating against the test set
tunes on it, and the final number stops meaning anything.

### Prediction point

State at intake the moment the prediction is made and what is known then. Every feature at step 4 is
checked against it. A column recorded during or after the outcome fails the check regardless of how
predictive it is.

## Evidence

The skill was written against observed behaviour rather than assumption. Four agents were given the
same bank marketing dataset, varying two things: whether the skill was present, and whether the user
was described as available.

| | Gagged ("cannot answer questions") | Invited ("I'm available") |
| --- | --- | --- |
| **No skill** | Ran to completion · no report file · no backport log · edited framework tests silently | Stopped and asked |
| **With skill** | Ran to completion · report file written · backport log written · framework change logged | Stopped and asked |

What this established:

- **Both structural rules earn their place.** Report-to-file and the backport log: 0 of 2 without the
  skill, 1 of 1 with it under identical pressure.
- **The checkpoints are not what causes stopping.** Both invited agents stopped with and without the
  skill. Stopping tracks the invitation, so the original absolute rule was reworded to the
  conditional above.
- **Most good ML practice needs no enforcement.** Every agent, with or without the skill, dropped the
  leaking column, detected the chronological ordering and chose a temporal split. Rules for behaviour
  that already happens would be bloat.
- **Two competent agents produce different models.** Different feature sets, different results, both
  defensible — which is the argument for surfacing decisions rather than for policing technique.

Testing also found three real defects in #0: a test that ignored the configured separator, `predict`
returning labels without scores, and the absence of a validation split. The first two are fixed.

## Out of scope, deliberately

Clustering, cross-validation, three-way splits, additional model families, date-part extraction,
high-cardinality encoding, hyperparameter search, calibration, feature importance output.

Two candidates now have evidence behind them, having been hit during testing:

- `split.validation_size`, to carve a validation slice from the training portion.
- `data.sentinels`, to declare values like `pdays = -1` so they become missing before validation.

## Verification

1. #0 stays green: `scripts/verify.py` passes.
2. A fresh copy of #0 trains on `sample.csv` unchanged.
3. **Untested:** whether the skill is discovered automatically. Every test named the skill file
   explicitly, so discovery via the description field has not been exercised.

## Risks and trade-offs

| Risk | Assessment |
| --- | --- |
| Copies drift apart | Accepted; inherent to the model. D5 is the mitigation and depends on discipline |
| The skill guarantees the checkpoint, not the judgement | It makes Claude stop and ask; it does not make the recommendation correct |
| #0 has no worked example | The reasoning lives in project repos instead. Accepted to keep #0 clean |
| Skeleton gaps surface mid-project | Expected. The backport log turns them into recorded findings rather than silent workarounds |
| Evidence is one run per cell | Directional, not conclusive. Re-test when a rule looks wrong in practice |

## Open questions

1. **Where do project copies live?** Sibling directories, or their own repositories?
2. **Do project repos get remotes?** #0 is public; a project containing client data may not be.
