"""Run the submission on several instruments in ONE process, as the grader does.

Checks the thing the per-instrument tools cannot: that import plus all three
predict() calls together stay inside the time limit. Reads practice data
written by the starter kit's make_sandbox.py (one directory per instrument,
each holding respondents.parquet), and the matching schema files.

    python tests/full_run.py --data /path/to/_sandbox --schemas /path/to/data
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_eval as le  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--schemas", required=True)
    parser.add_argument("--submission", default=str(
        Path(__file__).resolve().parents[1] / "submission"))
    parser.add_argument("--phase", type=int, default=1, choices=(1, 2))
    parser.add_argument("--instruments", nargs="+",
                        default=["unicef", "world_bank", "unhcr"])
    args = parser.parse_args()

    began = time.time()
    spec = importlib.util.spec_from_file_location(
        "submission_main", Path(args.submission) / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_main"] = module
    spec.loader.exec_module(module)

    skills = []
    for name in args.instruments:
        schema = le.load_schema(Path(args.schemas) / (name + ".json"))
        frame = pd.read_parquet(
            Path(args.data) / name / "respondents.parquet").astype(object)
        items = [n for n, r in schema["items"].items()
                 if r["class"] in ("GIVEN", "PREDICT")]
        shown, visible, cells = le.mask(frame, items, schema, args.phase)
        start = time.time()
        vectors = module.predict(shown.copy(), json.loads(json.dumps(schema)))
        le.validate(vectors, cells, schema)
        reference = -le.log_score(
            le.crowd_marginal(visible, cells, schema), cells, schema)
        skill = 1.0 + le.log_score(vectors, cells, schema) / reference
        skills.append(skill)
        print("PASS  %-10s skill %.4f  (%.1f s)"
              % (name, skill, time.time() - start), flush=True)
    print("mean skill %.4f, whole run %.1f s"
          % (sum(skills) / len(skills), time.time() - began))


if __name__ == "__main__":
    main()
