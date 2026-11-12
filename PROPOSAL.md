# SimulacraBench: submission proposal

*Drafted 2026-11-12. Based on the public starter kit `SituatedEvals/public` at commit `aa03186` (2026-10-26). I could not open simulacrabench.org, the Codabench pages or the NeurIPS page from this environment, so the Codabench Terms tab, prizes and current leaderboard have not been checked.*

## 0. Deadlines (these drive the whole plan)

| | Dates (AoE) | Hard cutoff (UTC) | Budget | Feedback |
|---|---|---|---|---|
| Phase 1, Development | ends **Nov 13** | 2026-11-14 12:00 | 900 s for the whole run | 3 submissions/day, noised and rounded |
| Phase 2, Final | Nov 15–19 | opens 11-15 12:00, closes 11-20 12:00 | 3600 s | **1 successful submission**, exact score |

That leaves roughly **4–6 development submissions**, depending on when Codabench resets the daily quota. Failed submissions do not count against the quota, so a crash costs time but not a submission.

**Do first:** register on Codabench (org 4076), accept the Terms and form the team (each person can be on only one team).

## 1. What the task really is (from reading the code)

1. **A held-out row shows only its `GIVEN` block.** Every `PREDICT` cell of a held-out row is blank, so you never see any of that person's other answers. Each cell is scored on its own. The job is therefore to estimate **P(item j | GIVEN block)** for every item, with the probabilities calibrated. Consistency between items only matters through the skip logic.
2. **The real data exists only inside the grading container.** In Phase 1, `predict(frame, schema)` receives the real `TRAIN` rows with their answers. In Phase 2 it receives `TRAIN` plus `DEV`. Anything fitted to the real data has to be **fitted inside `predict()`, on the clock**. Offline you only have the synthetic sandbox (one random latent trait and random marginals), which tells you nothing about the real data. Offline work therefore builds a pipeline that fits and checks itself, not a model trained in advance.
3. **The data is small and the machine is large.** The visible rows are about 16k × 19 columns (UNICEF), 5.7k × 85 (World Bank) and 8.9k × 230 (UNHCR), around 50 MB in total, on one H100 with 16 GB RAM and 8 CPUs. Fitting a few hundred models per instrument takes minutes, so the time budget will not be the limit. Overfitting will be.
4. **Score = skill vs. the crowd marginal.** The crowd marginal scores 0 and a perfect prediction scores 1. The headline number is the plain mean over the three instruments. Every vector is renormalized and mixed with ε = 1e-3, so a confident miss costs about −6.9 nats. Calibration matters more than being sharp.
5. **The skip logic is the cheapest gain, and the instruments differ:**

| Instrument | PREDICT items | Gated | Parent is `GIVEN` | Max chain depth | Notes |
|---|---|---|---|---|---|
| UNICEF | 12 | 7 | 5 | 1 | `Q01` (age) is a GIVEN that is gated on country (not fielded in Egypt) |
| World Bank | 76 | 30 | **27** | 2 | Mostly ordinal bins (Big-Five scores, hours, wages), K up to 25 |
| UNHCR | 213 | 156 | 1 | **4** | Long chains of PREDICT parents |

   When the parent is `GIVEN`, the gate slot is known exactly (the "never asked" slot gets probability 0 or 1). This is free score on 27 World Bank items, and the crowd marginal does not take it. For UNHCR, P(asked) has to be carried down chains up to four deep.

## 2. Proposed method: a per-item ensemble that fits and checks itself inside the container

All of this runs inside `predict()`, once per instrument.

```
for each instrument:
  visible rows -> encode GIVEN (one-hot + ordinal codes; "never asked" as its own level)
  for each PREDICT item j (parents before children):
     split into: A_j = P(asked | GIVEN)   and   B_j = P(answer | asked, GIVEN)
     fit candidate models for B_j on the visible rows where item j was asked
     5-fold out-of-fold (OOF) log loss -> weights for mixing the candidates + temperature
     refit on all visible rows, predict for the held-out rows
  A_j: exact (0/1) if the parent is GIVEN, else from the parent's prediction (recursion)
       or a direct binary model; choose by OOF
  vector = [A_j * B_j , 1 - A_j]      (gate slot last, as the schema requires)
```

**Candidate models for each item** (each one is optional; the crowd marginal is always included):

| # | Candidate | Why |
|---|---|---|
| c0 | Crowd marginal (+½ counts) | The reference. Keeping it in the mix means an item can fall back to a skill near 0 |
| c1 | Dirichlet backoff over the top MI-ranked GIVEN columns (as in `baseline/external_model`, LLM off) | Strong when n is small and there are few features; already written |
| c2 | Multinomial logistic regression on one-hot GIVEN, L2 penalty tuned by CV (sklearn) | Well calibrated and fast; likely the main model |
| c3 | `HistGradientBoostingClassifier`, shallow trees, early stopping | Picks up interactions such as country × religion and asylum country × governorate |
| c4 | Multi-task MLP on the GPU: shared trunk over GIVEN, one softmax head per item (torch) | Shares signal across UNHCR's 213 related items |
| c5 | *(stretch)* Ordinal model or adjacent-bin smoothing for binned World Bank items | Bins are ordered; uses that order |
| c6 | *(stretch)* LLM prior as the Dirichlet base measure (the `external_model` pattern, Qwen2.5-1.5B via `models.txt`) | Only helps rare cells and strata; the README itself says it is weak against thousands of real counts |

**Mixing weights for each item:** choose weights on the simplex that minimize the OOF log loss over c0…c4, then apply one temperature. Because c0 is always available, a bad candidate can never pull an item below the crowd by much. This is what makes a single blind Final submission safe.

**Robustness (needed with only one Final shot):**
- Use only libraries that are already in the image (numpy, pandas, scipy, scikit-learn, torch), so `requirements.txt` stays empty and dependency builds cannot fail.
- Keep a wall-clock budget manager. Fit the cheap candidates first, add expensive ones only while time remains (target: finish in under 50% of the budget), and stop cleanly with whatever is ready.
- Wrap the fit for each item in `try/except` and fall back to c0. `predict()` must never raise.
- Always output vectors that are finite, positive and the right width, in schema-key order (not `frame.columns` order), with the gate slot last.
- Fix the seeds and avoid threads that make results non-deterministic.

## 3. Using the development leaderboard well

- Feedback per instrument is fairly coarse. The Laplace noise is small (scale ≈ 6.9 / (n_dev · U · 10), about 0.001 on the uniform scale). The rounding to 0.01 on the uniform scale is larger, and it is stretched by U/R when converted to skill. Differences of a few hundredths are within this coarseness.
- Treat the leaderboard as a check that the pipeline works on real data and as a way to A/B one or two **big** decisions. Do not tune on it. The tuning happens through the in-container CV.
- Proposed submissions:
  1. **v1 (today):** c0 + exact gates + c1 + c2 with OOF mixing. Expect a clearly positive score on World Bank from the gates alone.
  2. **v2:** add c3 (trees).
  3. **v3:** add c4 (multi-task MLP). Keep whichever of v2/v3 wins on the mean, unless the gap is within the rounding.
  4. **Spare 1–2:** an ablation (gates only, or c2 only) to confirm where the gain comes from, or a fix for a failure.

## 4. Work plan

| When (UTC) | Task |
|---|---|
| Nov 12 | Register on Codabench and accept the Terms. Copy `baseline/external_model` to `submission/`, remove the LLM, add OOF mixing + c2. Pass `score.py` on all three sandboxes at `--phase 1` and `--phase 2`, and pass `tools/check_submission_zip.py`. **Submit v1.** |
| Nov 13 | Add c3 and c4 behind the time budget manager. Profile on the sandbox at real size (the sandbox has the real row counts). Submit v2 and v3. |
| Nov 14 (before 12:00) | Last development submission if a quota slot is left. Freeze the model. |
| Nov 14–15 | Hardening: make every candidate fail on purpose and confirm the fallback; time the run with `--phase 2` (more rows, 3600 s); test with `--docker` (16 GB / 8 CPU caps). Write the disclosure of models and data used (the FAQ asks for one). |
| Nov 15–16 | **Submit the Final early.** A failed run does not count, so submitting early leaves days to retry. |
| Nov 17–19 | Buffer only. |

## 5. Risks

| Risk | Mitigation |
|---|---|
| Overfitting thousands of small cells (UNHCR items deep in a gate chain have few asked rows) | Strong regularization; OOF mixing toward c0; a minimum asked count before an item gets anything beyond c1 |
| Wrong vector order or width (instant failure) | Build vectors from `schema["items"]` order; unit test against the three real schemas plus `check_submission_zip.py` |
| Timeout or OOM in Final (more rows) | Budget manager; rehearse with `--phase 2 --docker` |
| Sandbox results mislead | Use the sandbox only for correctness and timing, never for model choice |
| Rules or prizes not verified (site unreachable from here) | Read the Codabench Terms tab before submitting |

## Appendix: what was reviewed
- `README.md`, `FAQ.md`, `config.yml`, and the three baselines (`marginal_counts`, `bundled_artifact`, `external_model`)
- `score.py` (row selection, privacy and noise, reference scores), `make_sandbox.py`, `tests/`
- Schemas `data/{unicef,world_bank,unhcr,sample}.json`
- Not run: the starter-kit scripts. Running downloaded code was blocked in this session, so the sandbox scores have not been reproduced.
- Not read: simulacrabench.org and Codabench (network-blocked); the third-party repo `rono-labs/surveycast` and the GitHub issue `Luminous-Dynamics/symtropy#4541` showed up in search results only.
