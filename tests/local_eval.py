"""Local stand-in for the organizers' grader.

Builds synthetic respondents from a real schema file (items, options, skip
logic, split sizes), hides the PREDICT block of the held-out rows, calls the
submission's predict(), validates the output against the rules in the starter
kit README, and reports skill (0 = crowd marginal, 1 = perfect).

The answers are invented, so the score says nothing about the real data. Use
it to check that the code runs, returns legal vectors, and stays in budget.

    python tests/local_eval.py --schema path/to/unicef.json --phase 1
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GATED = "NA_GATED"
FLOOR = 1e-3


def load_schema(path):
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    schema.setdefault("gated_value", GATED)
    return schema


def options_for(schema, name):
    record = schema["items"][name]
    values = list(record["values"])
    if record.get("gate"):
        values.append(schema["gated_value"])
    return values


def synthesize(schema, seed=0, given_effect=1.0, trait_effect=1.5):
    """Invent respondents: answers depend on a hidden trait and on GIVEN."""
    rng = np.random.default_rng(seed)
    records = schema["items"]
    items = [n for n, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in items if records[n]["class"] == "GIVEN"]
    n = schema["dataset"]["n_rows"]

    def depth(name):
        level = 0
        while records[name].get("gate"):
            name = records[name]["gate"]["parent"]
            level += 1
        return level

    trait = rng.normal(size=n)
    data = {}
    order = sorted(items, key=lambda name: (records[name]["class"] != "GIVEN",
                                            depth(name)))
    for name in order:
        values = list(records[name]["values"])
        logits = np.log(rng.dirichlet(np.full(len(values), 0.9)) + 1e-6)
        logits = np.tile(logits, (n, 1))
        logits += np.outer(trait, rng.normal(scale=trait_effect, size=len(values)))
        if records[name]["class"] == "PREDICT" and given:
            for parent in rng.choice(given, size=min(2, len(given)), replace=False):
                levels = pd.Categorical(data[parent]).codes
                effect = rng.normal(scale=given_effect,
                                    size=(levels.max() + 1, len(values)))
                logits += effect[levels]
        gumbel = -np.log(-np.log(rng.uniform(size=logits.shape)))
        answer = np.array(values, dtype=object)[np.argmax(logits + gumbel, axis=1)]
        gate = records[name].get("gate")
        if gate and gate["parent"] in data:
            asked = np.isin(data[gate["parent"]], list(gate["observed_if"]))
            answer = np.where(asked, answer, schema["gated_value"])
        data[name] = answer.astype(object)

    frame = pd.DataFrame({name: data[name] for name in items}, dtype=object)
    frame.insert(0, "respondent_id", ["r%06d" % i for i in range(n)])
    split = schema["split"]
    frame["role"] = (["TRAIN"] * split["n_train"] + ["DEV"] * split["n_dev"]
                     + ["TEST"] * split["n_test"])
    return frame, items


def mask(frame, items, schema, phase):
    visible_roles, hidden_role = {1: (("TRAIN",), "DEV"),
                                  2: (("TRAIN", "DEV"), "TEST")}[phase]
    scored = [n for n in items if schema["items"][n]["class"] == "PREDICT"]
    visible = frame[frame["role"].isin(visible_roles)][["respondent_id"] + items]
    hidden = frame[frame["role"] == hidden_role][["respondent_id"] + items]
    blanked = hidden.copy()
    blanked[scored] = np.nan
    shown = pd.concat([visible, blanked], ignore_index=True)
    cells = [(item, hidden[item].iloc[row])
             for row in range(len(hidden)) for item in items if item in scored]
    return shown, visible, cells


def validate(vectors, cells, schema):
    assert isinstance(vectors, (list, tuple)), "return value must be a list"
    assert len(vectors) == len(cells), (
        "returned %d vectors for %d blank cells" % (len(vectors), len(cells)))
    for index, (vector, (item, _)) in enumerate(zip(vectors, cells)):
        p = np.asarray(vector, dtype=float)
        width = len(options_for(schema, item))
        assert p.shape == (width,), "vector %d (%s): wrong width" % (index, item)
        assert np.isfinite(p).all(), "vector %d (%s): not finite" % (index, item)
        assert (p >= 0).all(), "vector %d (%s): negative" % (index, item)
        assert p.sum() > 0, "vector %d (%s): all zero" % (index, item)


def log_score(vectors, cells, schema):
    total = 0.0
    index_of = {}
    for vector, (item, truth) in zip(vectors, cells):
        if item not in index_of:
            index_of[item] = {v: i for i, v in enumerate(options_for(schema, item))}
        p = np.asarray(vector, dtype=float)
        p = p / p.sum()
        p = FLOOR + (1.0 - p.size * FLOOR) * p
        total += np.log(p[index_of[item][truth]])
    return total / len(cells)


def crowd_marginal(visible, cells, schema):
    cache = {}
    out = []
    for item, _ in cells:
        if item not in cache:
            counts = visible[item].value_counts()
            weights = np.array([counts.get(o, 0) + 0.5
                                for o in options_for(schema, item)], float)
            cache[item] = weights / weights.sum()
        out.append(cache[item])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True)
    parser.add_argument("--submission", default=str(
        Path(__file__).resolve().parents[1] / "submission"))
    parser.add_argument("--phase", type=int, default=1, choices=(1, 2))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    frame, items = synthesize(schema, seed=args.seed)
    shown, visible, cells = mask(frame, items, schema, args.phase)

    began = time.time()
    spec = importlib.util.spec_from_file_location(
        "submission_main", Path(args.submission) / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_main"] = module
    spec.loader.exec_module(module)
    # Shuffle the columns: the contract says frame.columns order may differ.
    shuffled = ["respondent_id"] + list(
        np.random.default_rng(1).permutation(items))
    vectors = module.predict(shown[shuffled].copy(), json.loads(json.dumps(schema)))
    seconds = time.time() - began

    validate(vectors, cells, schema)
    reference = -log_score(crowd_marginal(visible, cells, schema), cells, schema)
    skill = 1.0 + log_score(vectors, cells, schema) / reference
    print("PASS  %s  phase %d  cells %d  skill %.4f  (%.1f s)"
          % (Path(args.schema).stem, args.phase, len(cells), skill, seconds))


if __name__ == "__main__":
    main()
