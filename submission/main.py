"""SimulacraBench submission, v1.

For every held-out respondent only the GIVEN block is visible, so each PREDICT
item is modelled as P(answer | GIVEN). Everything is fitted inside predict()
on the visible rows, because the real data only exists in the grader.

Per item:
  1. Skip logic. Rows whose GIVEN answers already decide that the item was
     never asked form their own stratum and get that stratum's observed
     distribution (in practice: all mass on the gate slot).
  2. On the remaining rows, several candidates are fitted and scored out of
     fold:
       c0  crowd marginal
       c1  count backoff over the two most informative GIVEN columns
       c2  multinomial logistic regression on one-hot GIVEN, three L2 strengths
  3. The candidates are mixed with per-item weights chosen on the out-of-fold
     log loss. c0 is always in the mix, so an item can fall back to the crowd.

If anything fails, predict() returns the crowd marginal instead of raising.
"""

import os
import time

_T0 = time.time()

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # torch missing: the logistic candidates are skipped
    torch = None

# Seconds for the whole run (import + all three predict() calls). The hosted
# limit is 900 s in Development; stay well inside it.
TOTAL_BUDGET = float(os.environ.get("SB_BUDGET", "600"))
N_INSTRUMENTS = 3
N_FOLDS = 5
MIN_TRAIN_ROWS = 50
BACKOFF_ALPHA = 10.0
BACKOFF_COLUMNS = 2
L2_STRENGTHS = (1e-3, 1e-4, 1e-5)
SEED = 0
VERBOSE = os.environ.get("SB_VERBOSE", "1") == "1"

_calls = 0


def _log(message):
    if VERBOSE:
        print("[v1 %6.1fs] %s" % (time.time() - _T0, message), flush=True)


# ---------------------------------------------------------------- encoding --
def _options(schema, item):
    record = schema["items"][item]
    values = list(record["values"])
    if record.get("gate"):
        values.append(schema["gated_value"])
    return values


def _codes(column, options):
    """Answers as indices into `options`; -1 for a blank or unknown value."""
    lookup = {}
    for index, option in enumerate(options):
        lookup.setdefault(option, index)
        lookup.setdefault(str(option), index)
    out = np.full(len(column), -1, dtype=np.int64)
    for row, value in enumerate(column):
        try:
            code = lookup.get(value)
            if code is None and value is not None and value == value:
                code = lookup.get(str(value))
        except TypeError:
            code = None
        if code is not None:
            out[row] = code
    return out


def _smoothed(counts, extra=None):
    weights = np.asarray(counts, float) + 0.5
    if extra is not None:
        weights = weights + extra
    return weights / weights.sum()


# ------------------------------------------------------ fallback: marginal --
def _marginal_predict(frame, schema):
    """The crowd marginal. Used when there is nothing to fit, or on failure."""
    items = [name for name, record in schema["items"].items()
             if record["class"] in ("GIVEN", "PREDICT")]
    marginals = {}
    for item in items:
        options = _options(schema, item)
        codes = _codes(frame[item].to_numpy(dtype=object), options)
        counts = np.bincount(codes[codes >= 0], minlength=len(options))
        marginals[item] = _smoothed(counts)
    blank = frame[items].isna().to_numpy()
    out = []
    for row in np.flatnonzero(blank.any(axis=1)):
        for column in np.flatnonzero(blank[row]):
            out.append(marginals[items[column]].tolist())
    return out


# ------------------------------------------------------------ candidate c1 --
def _mutual_information(x, y, nx, ny):
    both = (x >= 0) & (y >= 0)
    x, y = x[both], y[both]
    if x.size < 8:
        return 0.0
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny) / x.size
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def _backoff(y_fit, g_fit, g_eval, cards, width):
    """Marginal -> best GIVEN column -> best two columns, with pseudo-counts.

    y_fit    codes of one item on the fitting rows (all >= 0)
    g_fit    [rows, n_given] GIVEN codes on the fitting rows
    g_eval   [rows, n_given] GIVEN codes on the rows to predict
    """
    base = _smoothed(np.bincount(y_fit, minlength=width))
    running = np.tile(base, (g_eval.shape[0], 1))
    if y_fit.size < 8:
        return running
    scores = [_mutual_information(g_fit[:, g], y_fit, cards[g], width)
              for g in range(g_fit.shape[1])]
    ladder = np.argsort(scores)[::-1][:BACKOFF_COLUMNS]
    key_fit = np.zeros(y_fit.size, dtype=np.int64)
    key_eval = np.zeros(g_eval.shape[0], dtype=np.int64)
    size = 1
    for g in ladder:
        # One extra level per column for "unknown", so -1 never collides.
        card = cards[g] + 1
        key_fit = key_fit * card + (g_fit[:, g] + 1)
        key_eval = key_eval * card + (g_eval[:, g] + 1)
        size *= card
        table = np.bincount(key_fit * width + y_fit,
                            minlength=size * width).reshape(size, width)
        counts = table[key_eval].astype(float)
        running = ((counts + BACKOFF_ALPHA * running)
                   / (counts.sum(axis=1, keepdims=True) + BACKOFF_ALPHA))
    return running


# ------------------------------------------------------------ candidate c2 --
class _Linear:
    """Multinomial logistic regression for all items at once.

    One weight matrix maps the one-hot GIVEN block to a flat vector of slots;
    each item owns a contiguous run of slots and has its own softmax.
    """

    def __init__(self, widths, device):
        self.widths = list(widths)
        self.offsets = np.concatenate([[0], np.cumsum(self.widths)]).astype(int)
        self.n_slots = int(self.offsets[-1])
        self.device = device
        slot_item = np.repeat(np.arange(len(self.widths)), self.widths)
        self.slot_item = torch.as_tensor(slot_item, device=device)
        self.offsets_t = torch.as_tensor(self.offsets[:-1], device=device)

    def _log_norm(self, logits):
        peak = logits.max(dim=1, keepdim=True).values
        sums = torch.zeros(logits.shape[0], len(self.widths),
                           device=logits.device, dtype=logits.dtype)
        sums.index_add_(1, self.slot_item, torch.exp(logits - peak))
        return torch.log(sums.clamp_min(1e-30)) + peak

    def fit(self, x, y, l2, bias):
        """x [rows, features]; y [rows, items] codes, -1 = ignore the cell."""
        x = torch.as_tensor(x, device=self.device)
        y = torch.as_tensor(y, device=self.device)
        valid = (y >= 0).to(x.dtype)
        target = y.clamp_min(0) + self.offsets_t
        weight = torch.zeros(x.shape[1], self.n_slots, device=self.device,
                             dtype=x.dtype, requires_grad=True)
        offset = torch.as_tensor(bias, device=self.device,
                                 dtype=x.dtype).clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([weight, offset], lr=1.0, max_iter=200,
                                      history_size=20,
                                      line_search_fn="strong_wolfe")
        rows = float(x.shape[0])

        def closure():
            optimizer.zero_grad()
            logits = x @ weight + offset
            picked = logits.gather(1, target)
            loss = ((self._log_norm(logits) - picked) * valid).sum() / rows
            loss = loss + l2 * (weight ** 2).sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        self.weight, self.offset = weight.detach(), offset.detach()
        return self

    def predict(self, x):
        """Probabilities as one flat [rows, n_slots] array."""
        with torch.no_grad():
            x = torch.as_tensor(x, device=self.device)
            logits = x @ self.weight + self.offset
            log_p = logits - self._log_norm(logits)[:, self.slot_item]
            return torch.exp(log_p).double().cpu().numpy()


# ---------------------------------------------------------------- stacking --
def _mixture_weights(true_probs):
    """Weights on the simplex that maximise the out-of-fold log likelihood.

    true_probs [cells, candidates]: what each candidate gave the real answer.
    Candidate 0 is the crowd marginal and gets a small head start, so an item
    with little data stays close to it.
    """
    cells, candidates = true_probs.shape
    prior = np.zeros(candidates)
    prior[0] = 2.0
    weights = np.full(candidates, 1.0 / candidates)
    if cells == 0:
        return prior / prior.sum()
    probs = np.clip(true_probs, 1e-12, None)
    for _ in range(200):
        share = probs * weights
        share /= share.sum(axis=1, keepdims=True)
        updated = (share.sum(axis=0) + prior) / (cells + prior.sum())
        if np.abs(updated - weights).max() < 1e-6:
            weights = updated
            break
        weights = updated
    return weights


# -------------------------------------------------------------------- core --
def _predict_core(frame, schema, deadline):
    records = schema["items"]
    items = [n for n, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [n for n in items if records[n]["class"] == "GIVEN"]
    targets = [n for n in items if records[n]["class"] == "PREDICT"]
    options = {n: _options(schema, n) for n in items}
    widths = [len(options[n]) for n in targets]
    n_rows = len(frame)

    codes = {n: _codes(frame[n].to_numpy(dtype=object), options[n])
             for n in items}
    blank = frame[items].isna().to_numpy()
    hidden_rows = np.flatnonzero(blank.any(axis=1))
    if hidden_rows.size == 0:
        return []
    train_rows = np.flatnonzero(~blank.any(axis=1))
    if train_rows.size < MIN_TRAIN_ROWS or not targets:
        return _marginal_predict(frame, schema)

    y_all = np.stack([codes[n] for n in targets], axis=1)
    if given:
        g_all = np.stack([codes[n] for n in given], axis=1)
    else:
        g_all = np.zeros((n_rows, 0), dtype=np.int64)
    cards = [len(options[n]) for n in given]

    # ---- skip logic: rows where GIVEN already says "never asked" ---------
    def depth(name):
        level, seen = 0, set()
        while records.get(name, {}).get("gate") and name not in seen:
            seen.add(name)
            name = records[name]["gate"]["parent"]
            level += 1
        return level

    decided = {}
    for name in sorted(targets, key=depth):
        gate = records[name].get("gate")
        flag = np.zeros(n_rows, dtype=bool)
        if gate:
            parent, allowed = gate["parent"], list(gate["observed_if"])
            if parent in given:
                opens = [i for i, v in enumerate(options[parent]) if v in allowed]
                flag = (codes[parent] >= 0) & ~np.isin(codes[parent], opens)
            elif parent in decided and schema["gated_value"] not in allowed:
                flag = decided[parent]
        decided[name] = flag
    decided_all = np.stack([decided[n] for n in targets], axis=1)

    # Cells that take part in fitting: a known answer, on an undecided row.
    y_fit_all = np.where(decided_all, -1, y_all)

    # ---- folds over the visible rows -------------------------------------
    rng = np.random.default_rng(SEED)
    n_folds = max(2, min(N_FOLDS, train_rows.size // 10))
    fold_of = np.full(n_rows, -1, dtype=np.int64)
    fold_of[rng.permutation(train_rows)] = np.arange(train_rows.size) % n_folds
    # Each split: (rows to fit on, rows to predict). The last one is the real
    # fit: all visible rows -> hidden rows.
    splits = [(np.flatnonzero((fold_of >= 0) & (fold_of != f)),
               np.flatnonzero(fold_of == f)) for f in range(n_folds)]
    splits.append((train_rows, hidden_rows))

    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(int)
    n_slots = int(offsets[-1])

    def empty():
        """Flat probabilities for every row: filled on train (out of fold) and
        on hidden rows (full fit)."""
        return np.zeros((n_rows, n_slots))

    candidates, names = [], []

    # ---- c0: crowd marginal within the undecided stratum -----------------
    c0 = empty()
    for fit, evaluate in splits:
        for j in range(len(targets)):
            y = y_fit_all[fit, j]
            c0[evaluate, offsets[j]:offsets[j + 1]] = _smoothed(
                np.bincount(y[y >= 0], minlength=widths[j]))
    candidates.append(c0)
    names.append("marginal")

    # ---- c1: count backoff -----------------------------------------------
    try:
        if given and time.time() < deadline:
            c1 = empty()
            for fit, evaluate in splits:
                for j in range(len(targets)):
                    y = y_fit_all[fit, j]
                    keep = y >= 0
                    c1[evaluate, offsets[j]:offsets[j + 1]] = _backoff(
                        y[keep], g_all[fit][keep], g_all[evaluate], cards,
                        widths[j])
            candidates.append(c1)
            names.append("backoff")
    except Exception as error:
        _log("backoff skipped: %s" % type(error).__name__)

    # ---- c2: logistic regression, several L2 strengths -------------------
    if torch is not None and given:
        try:
            torch.manual_seed(SEED)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            x_all = np.zeros((n_rows, int(sum(cards))), dtype=np.float32)
            start = 0
            for g, card in enumerate(cards):
                known = np.flatnonzero(g_all[:, g] >= 0)
                x_all[known, start + g_all[known, g]] = 1.0
                start += card
            x_all = torch.as_tensor(x_all, device=device)
            last_cost = 0.0
            for l2 in L2_STRENGTHS:
                began = time.time()
                if began + 1.5 * last_cost > deadline:
                    _log("logistic l2=%g skipped: out of time" % l2)
                    break
                table = empty()
                for fit, evaluate in splits:
                    bias = np.log(np.clip(c0[evaluate[0]], 1e-6, None))
                    model = _Linear(widths, device).fit(
                        x_all[fit], y_fit_all[fit], l2, bias.astype(np.float32))
                    table[evaluate] = model.predict(x_all[evaluate])
                last_cost = time.time() - began
                if np.isfinite(table[train_rows]).all() and \
                        np.isfinite(table[hidden_rows]).all():
                    candidates.append(table)
                    names.append("logistic(l2=%g)" % l2)
        except Exception as error:
            _log("logistic skipped: %s" % type(error).__name__)

    # ---- mix per item on the out-of-fold log loss ------------------------
    result = {}
    total_mix, total_base, total_cells = 0.0, 0.0, 0
    for j, name in enumerate(targets):
        lo, hi = offsets[j], offsets[j + 1]
        y = y_fit_all[train_rows, j]
        rows = train_rows[y >= 0]
        picked = y[y >= 0]
        true_probs = np.stack(
            [c[rows, lo + picked] for c in candidates], axis=1)
        weights = _mixture_weights(true_probs)
        mixed = sum(w * c[hidden_rows, lo:hi]
                    for w, c in zip(weights, candidates))

        # Rows the skip logic decides: that stratum's own distribution, with
        # a prior on the gate slot (the last one) in case it is empty.
        stuck = decided_all[hidden_rows, j]
        if stuck.any():
            seen = y_all[train_rows, j][decided_all[train_rows, j]]
            extra = np.zeros(widths[j])
            extra[-1] = 5.0
            mixed[stuck] = _smoothed(
                np.bincount(seen[seen >= 0], minlength=widths[j]), extra)
        result[name] = mixed

        if rows.size:
            total_mix += np.log(np.clip(true_probs @ weights, 1e-12, None)).sum()
            total_base += np.log(np.clip(true_probs[:, 0], 1e-12, None)).sum()
            total_cells += rows.size

    if total_cells and total_base < 0:
        _log("%d items, %d visible rows, candidates: %s; out-of-fold skill on "
             "undecided cells ~ %.4f"
             % (len(targets), train_rows.size, ", ".join(names),
                1.0 - total_mix / total_base))

    # ---- one vector per blank cell, canonical order ----------------------
    given_marginal = {n: _smoothed(np.bincount(codes[n][codes[n] >= 0],
                                               minlength=len(options[n])))
                      for n in given}
    out = []
    for position, row in enumerate(hidden_rows):
        for column in np.flatnonzero(blank[row]):
            name = items[column]
            vector = (result[name][position] if name in result
                      else given_marginal[name])
            vector = np.nan_to_num(np.asarray(vector, float), nan=0.0,
                                   posinf=0.0, neginf=0.0).clip(min=0.0)
            if vector.sum() <= 0:
                vector = np.ones_like(vector)
            out.append((vector / vector.sum()).tolist())
    return out


def predict(frame, schema):
    """One probability vector per blank cell, in canonical order."""
    global _calls
    _calls += 1
    remaining = TOTAL_BUDGET - (time.time() - _T0)
    share = remaining / max(1, N_INSTRUMENTS - _calls + 1)
    deadline = time.time() + max(0.0, share)
    try:
        return _predict_core(frame, schema, deadline)
    except Exception as error:
        _log("falling back to the crowd marginal: %s" % type(error).__name__)
        return _marginal_predict(frame, schema)
