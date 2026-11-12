# SimulacraBench submission

- `PROPOSAL.md`: the plan.
- `submission/main.py`: the code that gets uploaded (v1).
- `tests/local_eval.py`: a local stand-in for the grader, on synthetic answers.

Build the upload:

```bash
cd submission && zip -r ../dist/submission_v1.zip . -x '__pycache__/*'
```

Check it with the organizers' tools (from a clone of `SituatedEvals/public`):

```bash
python tools/check_submission_zip.py /path/to/dist/submission_v1.zip
```

Local smoke test (schemas come from the starter kit's `data/`):

```bash
python tests/local_eval.py --schema /path/to/public/data/unhcr.json --phase 1
```
