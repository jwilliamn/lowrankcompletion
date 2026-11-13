"""SimulacraBench submission, v5.

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
       r   sharp count tables with gate recursion, the method of the starter
           kit's stronger baseline: P(answer | asked) from counts within the
           two or three most informative GIVEN items (light smoothing), and
           P(asked) worked down the gate chain from the parent's prediction
       t   gradient-boosted trees, one model per item (LightGBM), in two
           variants: on the GIVEN block alone, and on the GIVEN block plus
           what the respondent's nearest visible profile matches answered on
           this item and on its gate parent (their most common answer, how
           unanimous they were, how close the match is; kept apart for
           matches in the same and in another level of the "twin" item, e.g.
           the other survey round). Trees are sharp where the answer is
           fully determined, and learn how far to trust a match
       c2  multinomial logistic regression on one-hot GIVEN, several L2 strengths
       c2b the same with an "ordered answers" head: instead of one free weight
           per answer option, each item gets a shift and a spread along its
           option order plus a skip-slot term; far fewer parameters, meant
           for binned scores, hours and wages
       c2c the same on GIVEN plus its interactions with the two GIVEN items
           that say most about the answers (for example survey round or
           country), i.e. a separate set of effects per round / country
       c3  multi-task neural net (shared trunk, one softmax head per item),
           read out at several training lengths; picks up interactions
       c4  the same net, also fed what the respondent's closest profile
           matches answered on every item (their votes, split by whether the
           match differs on the "twin" GIVEN item, e.g. the other survey
           round). The net learns how a match's answers translate into this
           respondent's answers, instead of copying them
  3. The candidates are mixed with per-item weights chosen on the out-of-fold
     log loss. c0 is always in the mix, so an item can fall back to the crowd.

  4. Profile matching. Visible respondents whose GIVEN profile is identical or
     nearly identical to the target's (for example the same household in
     another survey round) vote with their own answers; where there is no
     close match the mixed model from step 3 is used unchanged. A second pass
     ignores the one GIVEN item whose mismatch matters least, found out of
     fold. How far to trust the matches is again chosen per item out of fold.

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

try:
    import lightgbm
except Exception:  # not installed: the tree candidates are skipped
    lightgbm = None
import pickle
import signal
import threading

# Seconds for the whole run (import + all three predict() calls). The hosted
# limit is 900 s in Development; stay well inside it.
TOTAL_BUDGET = float(os.environ.get("SB_BUDGET", "660"))
# The Final phase allows 3600 s. It is recognised by the DEV respondents
# being visible (more visible rows than the schema's n_train).
FINAL_BUDGET = float(os.environ.get("SB_FINAL_BUDGET", "2400"))
N_INSTRUMENTS = 3
N_FOLDS = 5
MIN_TRAIN_ROWS = 50
BACKOFF_ALPHA = 10.0
BACKOFF_COLUMNS = 2
L2_STRENGTHS = (1e-2, 1e-3, 1e-4, 1e-5)
TILT_L2 = (1e-3, 1e-4)
# Profile matching: weight exp(-beta * mismatches); matches further away than
# `reach` mismatches count for less than the fallback model.
MATCH_BETA = 6.0
MATCH_REACH = (0.5, 0.75, 1.0, 1.5, 2.5, 3.5)
MATCH_DROP_REACH = (0.5, 0.75, 1.0, 1.5)
MATCH_DROP_SAMPLE = 1500
INTERACTION_COLUMNS = 2
INTERACTION_MAX_LEVELS = 12
INTERACTION_L2 = (1e-3, 1e-4)
NEIGHBOUR_SEEDS = (0, 1)
POWER_GRID = (0.85, 0.92, 1.08, 1.16, 1.25, 1.35, 1.5)
POWER_GRID_SURE = (0.7, 0.85, 1.2, 1.5, 2.0, 3.0, 5.0)   # for confident rows
POWER_SURE = 0.9          # a row is "confident" above this top probability
POWER_MIN_CELLS = 200
RECURSION_ALPHA = 2.0     # pseudo-counts per level of the sharp count tables
RECURSION_DEPTHS = (2, 3)  # GIVEN items per table
GBDT_ROUNDS = (40, 100)   # boosting rounds at which the trees are read out
GBDT_MAX_DISTANCE = 3     # profile matches further away than this are ignored
MLP_CHECKPOINTS = (4, 8, 16, 32)     # epochs at which the net is read out
MLP_SEEDS = (0, 1, 2)
MLP_HIDDEN = 512
MLP_DROPOUT = 0.15
SEED = 0
VERBOSE = os.environ.get("SB_VERBOSE", "1") == "1"

_calls = 0


def _log(message):
    if VERBOSE:
        print("[v5 %6.1fs] %s" % (time.time() - _T0, message), flush=True)


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


def _recursion(g_all, y_all, splits, widths, cards, gates, depth, c0):
    """Sharp count tables with gate recursion.

    For each item, P(answer | asked) is counted among the visible respondents
    who were asked, within the `depth` GIVEN items that carry the most
    information about the answer, each level smoothed toward the coarser one
    with RECURSION_ALPHA pseudo-counts. P(asked) is 0 or 1 when the gate's
    parent is a GIVEN item, and otherwise the parent's own predicted
    probability of an answer that opens the gate.

    gates[j] is None, ("given", column, opening codes) or
    ("item", parent index, opening codes). Returns a table [rows, slots].
    """
    n_rows = g_all.shape[0]
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(int)
    table = np.zeros((n_rows, int(offsets[-1])))

    def level(j):
        seen, steps = set(), 0
        while gates[j] is not None and gates[j][0] == "item" and j not in seen:
            seen.add(j)
            j = gates[j][1]
            steps += 1
        return steps

    order = sorted(range(len(widths)), key=level)
    for fit, evaluate in splits:
        g_fit, g_eval = g_all[fit], g_all[evaluate]
        answer, asked = {}, {}
        for j in order:
            values = widths[j] - (0 if gates[j] is None else 1)
            y = y_all[fit, j]
            was_asked = (y >= 0) & (y < values)
            y_asked, g_asked = y[was_asked], g_fit[was_asked]
            running = np.full((len(evaluate), values), 1.0 / values)
            scores = [_mutual_information(g_asked[:, g], y_asked, cards[g],
                                          values)
                      for g in range(g_all.shape[1])]
            key_fit = np.zeros(y_asked.size, dtype=np.int64)
            key_eval = np.zeros(len(evaluate), dtype=np.int64)
            size = 1
            for g in [None] + list(np.argsort(scores)[::-1][:depth]):
                if g is not None:
                    card = cards[g] + 1
                    if size * card * values > 5_000_000:
                        break
                    key_fit = key_fit * card + (g_asked[:, g] + 1)
                    key_eval = key_eval * card + (g_eval[:, g] + 1)
                    size *= card
                counts = np.bincount(
                    key_fit * values + y_asked,
                    minlength=size * values).reshape(size, values)[key_eval]
                running = ((counts + RECURSION_ALPHA * running)
                           / (counts.sum(axis=1, keepdims=True)
                              + RECURSION_ALPHA))
            answer[j] = running
            if gates[j] is None:
                asked[j] = np.ones(len(evaluate))
                full = running
            else:
                kind, parent, opens = gates[j]
                if kind == "given":
                    asked[j] = np.isin(g_eval[:, parent], opens).astype(float)
                else:
                    asked[j] = asked[parent] * answer[parent][:, opens].sum(axis=1)
                share = asked[j][:, None]
                full = np.concatenate([share * running, 1.0 - share], axis=1)
            lo, hi = offsets[j], offsets[j + 1]
            table[evaluate, lo:hi] = 0.999 * full + 0.001 * c0[evaluate, lo:hi]
    return table


# ------------------------------------------------------------ candidate c2 --
class _Linear:
    """Multinomial logistic regression for all items at once.

    One weight matrix maps the one-hot GIVEN block to a flat vector of slots;
    each item owns a contiguous run of slots and has its own softmax.
    """

    def __init__(self, widths, device, basis=None):
        # basis [slots, outputs]: when given, the model predicts `outputs`
        # numbers per respondent and logits are their combination.
        self.basis = basis
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
        outputs = self.n_slots if self.basis is None else self.basis.shape[1]
        weight = torch.zeros(x.shape[1], outputs, device=self.device,
                             dtype=x.dtype, requires_grad=True)
        offset = torch.as_tensor(bias, device=self.device,
                                 dtype=x.dtype).clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([weight, offset], lr=1.0, max_iter=200,
                                      history_size=20,
                                      line_search_fn="strong_wolfe")
        rows = float(x.shape[0])

        def closure():
            optimizer.zero_grad()
            logits = self._logits(x, weight, offset)
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
            logits = self._logits(x, self.weight, self.offset)
            log_p = logits - self._log_norm(logits)[:, self.slot_item]
            return torch.exp(log_p).double().cpu().numpy()

    def _logits(self, x, weight, offset):
        scores = x @ weight
        if self.basis is not None:
            scores = scores @ self.basis.T
        return scores + offset


def _tilt_basis(widths, gated, device):
    """Basis for the "ordered answers" head: per item a shift along the
    option order, a spread term, and a term for the skip slot."""
    n_slots = int(sum(widths))
    basis = np.zeros((n_slots, 3 * len(widths)), dtype=np.float32)
    start = 0
    for j, width in enumerate(widths):
        values = width - (1 if gated[j] else 0)
        if values > 1:
            t = np.linspace(-1.0, 1.0, values)
            basis[start:start + values, 3 * j] = t
            basis[start:start + values, 3 * j + 1] = t ** 2 - np.mean(t ** 2)
        if gated[j]:
            basis[start + width - 1, 3 * j + 2] = 1.0
        start += width
    return torch.as_tensor(basis, device=device)


def _masked_loss(logits, target, valid, log_norm):
    picked = logits.gather(1, target)
    return ((log_norm(logits) - picked) * valid).sum() / logits.shape[0]


# -------------------------------------------------------- profile matching --
def _match(x_all, y_fit_all, splits, widths, smooth, n_given, device,
           drop=None, reaches=MATCH_REACH, only=None):
    """Let near-identical visible respondents vote, backed by `smooth`.

    A visible respondent's weight is exp(-MATCH_BETA * mismatching GIVEN
    answers). The fallback model `smooth` enters with the weight of a match
    at `reach` mismatches, so closer matches override it and farther ones do
    not. `drop` = (first, last) one-hot columns of a GIVEN item to ignore.
    `only` restricts the rows that are predicted. Returns one table
    [rows, slots] per value in `reaches`.
    """
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(int)
    n_slots = int(offsets[-1])
    slot_item = torch.as_tensor(np.repeat(np.arange(len(widths)), widths),
                                device=device)
    offsets_t = torch.as_tensor(offsets[:-1], device=device)
    tables = [np.zeros((x_all.shape[0], n_slots)) for _ in reaches]
    for fit, evaluate in splits:
        if only is not None:
            evaluate = np.intersect1d(evaluate, only)
            if evaluate.size == 0:
                continue
        y = torch.as_tensor(y_fit_all[fit], device=device)
        valid = (y >= 0).to(torch.float64)
        onehot = torch.zeros(len(fit), n_slots, device=device,
                             dtype=torch.float64)
        onehot.scatter_(1, y.clamp_min(0) + offsets_t, valid)
        x_fit = x_all[fit]
        for start in range(0, len(evaluate), 2048):
            chunk = evaluate[start:start + 2048]
            differ = n_given - x_all[chunk] @ x_fit.T
            if drop is not None:
                lo, hi = drop
                differ = differ - (1.0 - x_all[chunk][:, lo:hi]
                                   @ x_fit[:, lo:hi].T)
            weight = torch.exp(-MATCH_BETA * differ.double())
            top = weight @ onehot
            bottom = (weight @ valid)[:, slot_item]
            prior = torch.as_tensor(smooth[chunk], device=device)
            for index, reach in enumerate(reaches):
                alpha = float(np.exp(-MATCH_BETA * reach))
                tables[index][chunk] = (
                    (top + alpha * prior) / (bottom + alpha)).cpu().numpy()
    return tables


def _best_drop(x_all, y_fit_all, folds, widths, smooth, cards, device,
               train_rows, deadline):
    """The GIVEN item whose mismatch is best ignored when matching profiles,
    as one-hot (first, last) columns, or None if ignoring none is best.

    Judged out of fold on a sample of visible rows: exact matching on every
    GIVEN item but one, scored by the log likelihood of the real answers.
    """
    rng = np.random.default_rng(SEED + 1)
    sample = np.sort(rng.choice(train_rows, replace=False, size=min(
        MATCH_DROP_SAMPLE, train_rows.size)))
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(int)

    def score(drop):
        table = _match(x_all, y_fit_all, folds, widths, smooth, len(cards),
                       device, drop=drop, reaches=(0.5,), only=sample)[0]
        y = y_fit_all[sample]
        rows, cols = np.nonzero(y >= 0)
        probs = table[sample[rows], offsets[cols] + y[rows, cols]]
        return float(np.log(np.clip(probs, 1e-12, None)).sum())

    best, best_score = None, score(None)
    starts = np.concatenate([[0], np.cumsum(cards)]).astype(int)
    for g in range(len(cards)):
        if time.time() > deadline:
            break
        value = score((int(starts[g]), int(starts[g + 1])))
        if value > best_score + 1.0:
            best, best_score = (int(starts[g]), int(starts[g + 1])), value
    return best


# ------------------------------------------------------------ candidate c3 --
class _Net(torch.nn.Module if torch is not None else object):
    """Shared trunk with one softmax head per item. With `vote_width` set, a
    second, direct path maps the neighbours' votes on an item straight to
    that item's own answer slots (a small table per item), so reusing or
    translating a match's answer does not have to squeeze through the trunk.
    """

    def __init__(self, n_in, n_slots, slot_item, vote_width=0):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(n_in, MLP_HIDDEN), torch.nn.GELU(),
            torch.nn.Dropout(MLP_DROPOUT),
            torch.nn.Linear(MLP_HIDDEN, MLP_HIDDEN), torch.nn.GELU(),
            torch.nn.Dropout(MLP_DROPOUT),
            torch.nn.Linear(MLP_HIDDEN, n_slots))
        self.direct = None
        if vote_width:
            blocks = vote_width // n_slots
            same_item = (slot_item[:, None] == slot_item[None, :]).float()
            self.register_buffer("mask", same_item.repeat(1, blocks))
            self.direct = torch.nn.Parameter(torch.zeros(n_slots, vote_width))

    def forward(self, x, votes=None):
        logits = self.trunk(x)
        if self.direct is not None:
            logits = logits + votes @ (self.direct * self.mask).T
        return logits


def _mlp(x_all, y_fit_all, splits, widths, c0, device, seed, features=None):
    """Multi-task net: shared trunk over one-hot GIVEN, a softmax per item.

    The net is read out after each epoch count in MLP_CHECKPOINTS, and each
    read-out is its own candidate, so the mixing step chooses how long to
    train, per item. Returns one table [rows, slots] per checkpoint.
    `features`, when given, holds per split the tuple returned by
    _neighbour_features and switches on the direct path from votes.
    """
    helper = _Linear(widths, device)
    n_slots = helper.n_slots
    tables = [np.zeros((x_all.shape[0], n_slots)) for _ in MLP_CHECKPOINTS]
    for split_index, (fit, evaluate) in enumerate(splits):
        torch.manual_seed(1000 * seed + split_index)
        if features is None:
            x_fit, x_eval, v_fit, v_eval = x_all[fit], x_all[evaluate], None, None
        else:
            x_fit, x_eval, v_fit, v_eval = features[split_index]
        net = _Net(x_fit.shape[1], n_slots, helper.slot_item,
                   0 if v_fit is None else v_fit.shape[1]).to(device)
        with torch.no_grad():
            net.trunk[-1].weight.mul_(0.1)
            net.trunk[-1].bias.copy_(torch.as_tensor(
                np.log(np.clip(c0[evaluate[0]], 1e-6, None)),
                device=device, dtype=torch.float32))
        groups = [{"params": net.trunk.parameters(), "lr": 2e-3,
                   "weight_decay": 1e-2}]
        if net.direct is not None:
            groups.append({"params": [net.direct], "lr": 1e-2,
                           "weight_decay": 1e-3})
        optimizer = torch.optim.AdamW(groups)
        y = torch.as_tensor(y_fit_all[fit], device=device)
        valid = (y >= 0).to(torch.float32)
        target = y.clamp_min(0) + helper.offsets_t
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1000 * seed + split_index)
        for epoch in range(1, max(MLP_CHECKPOINTS) + 1):
            net.train()
            order = torch.randperm(len(fit), generator=generator).to(device)
            for start in range(0, len(fit), 256):
                rows = order[start:start + 256]
                optimizer.zero_grad()
                logits = net(x_fit[rows], None if v_fit is None else v_fit[rows])
                loss = _masked_loss(logits, target[rows], valid[rows],
                                    helper._log_norm)
                loss.backward()
                optimizer.step()
            if epoch in MLP_CHECKPOINTS:
                net.eval()
                with torch.no_grad():
                    logits = net(x_eval, v_eval)
                    log_p = logits - helper._log_norm(logits)[:, helper.slot_item]
                    tables[MLP_CHECKPOINTS.index(epoch)][evaluate] = (
                        torch.exp(log_p).double().cpu().numpy())
    return tables


# ------------------------------------------------------- trees and matches --
def _twin_item(x, g_all, train_rows, cards):
    """Index of the GIVEN item on which near-identical respondents most often
    differ: the item g with the most visible respondents that have another
    visible respondent identical on every GIVEN item except g."""
    rng = np.random.default_rng(SEED + 2)
    sample = rng.choice(train_rows, replace=False,
                        size=min(1500, train_rows.size))
    differ = len(cards) - x[sample] @ x[train_rows].T
    best, best_count = None, 0
    for g in range(len(cards)):
        other = g_all[sample, g][:, None] != g_all[train_rows, g][None, :]
        count = int((other & (differ < 1.5)).any(axis=1).sum())
        if count > best_count:
            best, best_count = g, count
    return best


def _profile_votes(x, g_all, y_all, train_rows, widths, twin):
    """What each respondent's nearest visible profile matches answered.

    For every respondent (visible or held out) the visible respondents with
    the fewest mismatching GIVEN answers are found, separately among those
    with the same and with another level of the twin item (the twin item
    itself is not counted as a mismatch). A visible respondent is never its
    own match. Per group: the distance, the number of matches at that
    distance, and for every item their most common answer and its share.
    """
    n_rows, n_given = g_all.shape
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(int)
    x_ref = x[train_rows]
    onehot = np.zeros((train_rows.size, int(offsets[-1])), dtype=np.float32)
    for j in range(len(widths)):
        y = y_all[train_rows, j]
        known = np.flatnonzero(y >= 0)
        onehot[known, offsets[j] + y[known]] = 1.0
    position = np.full(n_rows, -1, dtype=np.int64)
    position[train_rows] = np.arange(train_rows.size)
    names = ("same",) if twin is None else ("same", "other")
    out = {name: {"distance": np.full(n_rows, np.nan, dtype=np.float32),
                  "count": np.zeros(n_rows, dtype=np.float32),
                  "answer": np.full((n_rows, len(widths)), np.nan,
                                    dtype=np.float32),
                  "share": np.full((n_rows, len(widths)), np.nan,
                                   dtype=np.float32)} for name in names}
    for start in range(0, n_rows, 1024):
        rows = np.arange(start, min(n_rows, start + 1024))
        differ = n_given - x[rows] @ x_ref.T
        own = position[rows]
        differ[np.flatnonzero(own >= 0), own[own >= 0]] = 99.0
        if twin is None:
            groups = {"same": differ}
        else:
            same = g_all[rows, twin][:, None] == g_all[train_rows, twin][None, :]
            groups = {"same": np.where(same, differ, 99.0),
                      "other": np.where(same, 99.0, differ - 1.0)}
        for name, distance in groups.items():
            nearest = distance.min(axis=1)
            near = ((distance <= nearest[:, None] + 0.5)
                    & (nearest[:, None] <= GBDT_MAX_DISTANCE + 0.5))
            votes = near.astype(np.float32) @ onehot
            count = near.sum(axis=1)
            found = count > 0
            out[name]["distance"][rows] = np.where(found, nearest, np.nan)
            out[name]["count"][rows] = count
            for j in range(len(widths)):
                block = votes[:, offsets[j]:offsets[j + 1]]
                total = block.sum(axis=1)
                seen = total > 0
                out[name]["answer"][rows, j] = np.where(
                    seen, block.argmax(axis=1), np.nan)
                out[name]["share"][rows, j] = np.where(
                    seen, block.max(axis=1) / np.maximum(total, 1.0), np.nan)
    return out


def _forked(task, count, workers, deadline):
    """Run task(j) for j in range(count) in forked worker processes and
    return {j: result} for the ones that finished.

    Separate processes rather than threads, because training LightGBM models
    from several threads at once can crash. Results come back over pipes. If
    a worker dies or overruns, its unfinished items are simply missing.
    """
    pipes, pids = [], []
    for worker in range(workers):
        mine = list(range(worker, count, workers))
        if not mine:
            continue
        read_end, write_end = os.pipe()
        pid = os.fork()
        if pid == 0:                              # worker process
            try:
                os.close(read_end)
                with os.fdopen(write_end, "wb") as out:
                    for j in mine:
                        if time.time() > deadline:
                            break
                        pickle.dump((j, task(j)), out, protocol=4)
                        out.flush()
            except BaseException:
                pass
            finally:
                os._exit(0)
        os.close(write_end)
        pipes.append(read_end)
        pids.append(pid)

    results = {}

    def collect(descriptor):
        try:
            with os.fdopen(descriptor, "rb") as source:
                while True:
                    j, value = pickle.load(source)
                    results[j] = value
        except Exception:                         # EOFError when a worker ends
            pass

    readers = [threading.Thread(target=collect, args=(d,), daemon=True)
               for d in pipes]
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join(max(1.0, deadline + 30.0 - time.time()))
    for pid in pids:
        try:
            if any(reader.is_alive() for reader in readers):
                os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except Exception:
            pass
    return dict(results)


def _trees(g_all, y_fit_all, splits, widths, c0, cards, votes, parents,
           deadline, workers):
    """Gradient-boosted trees, one multiclass model per item and split.

    Inputs: the GIVEN block as ordered codes, plus (for items with five or
    more levels) the same codes again as unordered categories, so a tree can
    use either reading. With `votes`, also the nearest profile matches'
    answer on this item and on its gate parent, their share and distance.
    Returns one table [rows, slots] per entry of GBDT_ROUNDS and the share
    of items that finished in time (the rest keep the crowd marginal).
    """
    n_rows = g_all.shape[0]
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(int)
    tables = [c0.copy() for _ in GBDT_ROUNDS]
    ordered = np.where(g_all >= 0, g_all, np.nan).astype(np.float32)
    nominal = [g for g, card in enumerate(cards) if card >= 5]
    shared = [ordered, ordered[:, nominal]]
    categorical = list(range(ordered.shape[1],
                             ordered.shape[1] + len(nominal)))
    width = ordered.shape[1] + len(nominal)
    if votes:
        for group in votes.values():
            shared += [group["distance"][:, None], group["count"][:, None]]
            width += 2
    shared = np.concatenate(shared, axis=1)

    def task(j):
        """Out-of-fold and held-out predictions for item j: one array
        [rows, options] per entry of GBDT_ROUNDS, or None."""
        lo, hi = offsets[j], offsets[j + 1]
        columns, kinds = [shared], list(categorical)
        if votes:
            extra = width
            for group in votes.values():
                columns += [group["answer"][:, j:j + 1],
                            group["share"][:, j:j + 1]]
                kinds.append(extra)
                extra += 2
                if parents[j] is not None:
                    columns.append(group["answer"][:, parents[j]:parents[j] + 1])
                    kinds.append(extra)
                    extra += 1
        features = np.concatenate(columns, axis=1)
        out = [c0[:, lo:hi].astype(np.float32) for _ in GBDT_ROUNDS]
        for fit, evaluate in splits:
            y = y_fit_all[fit, j]
            keep = y >= 0
            classes = np.unique(y[keep])
            if keep.sum() < 40 or classes.size < 2:
                continue
            relabel = np.full(widths[j], -1, dtype=np.int64)
            relabel[classes] = np.arange(classes.size)
            settings = {"verbosity": -1, "max_bin": 63,
                        "min_data_per_group": 20, "cat_smooth": 5.0,
                        "feature_pre_filter": False, "num_threads": 1}
            data = lightgbm.Dataset(features[fit][keep], relabel[y[keep]],
                                    categorical_feature=kinds,
                                    free_raw_data=False, params=settings)
            booster = lightgbm.train(
                dict(settings, objective="multiclass",
                     num_class=int(classes.size), learning_rate=0.1,
                     num_leaves=15, min_data_in_leaf=15, bagging_fraction=0.8,
                     bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
                     seed=SEED),
                data, num_boost_round=max(GBDT_ROUNDS))
            for index, rounds in enumerate(GBDT_ROUNDS):
                probs = np.zeros((len(evaluate), widths[j]))
                probs[:, classes] = booster.predict(
                    features[evaluate], num_iteration=rounds).reshape(
                        len(evaluate), classes.size)
                # A trace of the crowd marginal, so an answer never seen in
                # the fitting rows is unlikely rather than impossible.
                out[index][evaluate] = (0.998 * probs
                                        + 0.002 * c0[evaluate, lo:hi])
        return out

    count = len(widths)
    try:
        results = _forked(task, count, workers, deadline)
    except Exception:                 # no fork here: run in this process
        results = {}
        for j in range(count):
            if time.time() > deadline:
                break
            results[j] = task(j)
    for j, out in results.items():
        if out is None:
            continue
        for index in range(len(GBDT_ROUNDS)):
            tables[index][:, offsets[j]:offsets[j + 1]] = out[index]
    return tables, len(results) / float(max(1, count))


# ------------------------------------------------- interactions, neighbours --
def _interaction_columns(g_all, y_fit_all, train_rows, cards, widths):
    """The GIVEN items (few levels only) that say most about the answers,
    by mutual information summed over items, corrected for level count."""
    scores = []
    for g, card in enumerate(cards):
        if card > INTERACTION_MAX_LEVELS or card < 2:
            scores.append(-np.inf)
            continue
        x = g_all[train_rows, g]
        total = 0.0
        for j, width in enumerate(widths):
            y = y_fit_all[train_rows, j]
            n = int(((x >= 0) & (y >= 0)).sum())
            if n < 50:
                continue
            total += (_mutual_information(x, y, card, width)
                      - (card - 1) * (width - 1) / (2.0 * n))
        scores.append(total)
    order = [g for g in np.argsort(scores)[::-1] if np.isfinite(scores[g])]
    return order[:INTERACTION_COLUMNS]


def _twin_column(x_all, train_rows, cards, device):
    """The GIVEN item on which near-identical respondents most often differ:
    the item g with the most visible respondents that have another visible
    respondent identical on every GIVEN item except g. Returns its one-hot
    (first, last) columns, or None."""
    rng = np.random.default_rng(SEED + 2)
    sample = rng.choice(train_rows, replace=False,
                        size=min(2000, train_rows.size))
    x_fit, x_sample = x_all[train_rows], x_all[sample]
    differ = len(cards) - x_sample @ x_fit.T
    starts = np.concatenate([[0], np.cumsum(cards)]).astype(int)
    best, best_count = None, 0
    for g in range(len(cards)):
        lo, hi = int(starts[g]), int(starts[g + 1])
        other = 1.0 - x_sample[:, lo:hi] @ x_fit[:, lo:hi].T
        hit = ((differ - other) < 0.5) & (other > 0.5)
        count = int(hit.any(dim=1).sum())
        if count > best_count:
            best, best_count = (lo, hi), count
    return best


def _neighbour_features(x_all, y_fit_all, fit, evaluate, widths, twin,
                        n_given, device):
    """Inputs for the neighbour-aware net.

    For every answer slot, the vote of the respondent's closest profile
    matches among the `fit` rows: exact matches; matches within about one
    mismatch; and matches that differ only on the twin item, kept apart by
    the respondent's own level of that item (so that "my match from round 1
    said X" and "my match from round 2 said X" are different inputs). Votes
    are on a log scale. A fit row never votes for itself.

    Returns (trunk inputs, votes) for the fit rows and for the evaluate rows.
    """
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(int)
    n_slots = int(offsets[-1])
    slot_item = torch.as_tensor(np.repeat(np.arange(len(widths)), widths),
                                device=device)
    offsets_t = torch.as_tensor(offsets[:-1], device=device)
    y = torch.as_tensor(y_fit_all[fit], device=device)
    valid = (y >= 0).to(torch.float32)
    onehot = torch.zeros(len(fit), n_slots, device=device)
    onehot.scatter_(1, y.clamp_min(0) + offsets_t, valid)
    x_fit = x_all[fit]
    near, far = float(np.exp(-MATCH_BETA * 0.5)), float(np.exp(-MATCH_BETA * 1.5))

    def scale(vote):
        return torch.log1p(100.0 * vote)

    def build(rows, own):
        trunk, votes = [], []
        for start in range(0, len(rows), 2048):
            x_rows = x_all[rows[start:start + 2048]]
            differ = n_given - x_rows @ x_fit.T
            if own:
                index = torch.arange(x_rows.shape[0], device=device)
                differ[index, start + index] = 1e3
            weight = torch.exp(-MATCH_BETA * differ)
            top = weight @ onehot
            bottom = (weight @ valid)[:, slot_item]
            blocks = [scale(top / (bottom + near)), scale(top / (bottom + far))]
            if twin is not None:
                lo, hi = twin
                other = 1.0 - x_rows[:, lo:hi] @ x_fit[:, lo:hi].T
                weight = torch.exp(-MATCH_BETA * (differ - other)) * other
                vote = scale((weight @ onehot)
                             / ((weight @ valid)[:, slot_item] + near))
                if hi - lo <= 4:
                    blocks += [vote * x_rows[:, level:level + 1]
                               for level in range(lo, hi)]
                else:
                    blocks.append(vote)
            votes.append(torch.cat(blocks, dim=1))
            trunk.append(torch.cat([x_rows, votes[-1]], dim=1))
        return torch.cat(trunk, dim=0), torch.cat(votes, dim=0)

    fit_trunk, fit_votes = build(fit, True)
    eval_trunk, eval_votes = build(evaluate, False)
    return fit_trunk, eval_trunk, fit_votes, eval_votes


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


def _best_power(probs, picked, grid):
    """Sharpening exponent for a set of mixed predictions.

    A weighted average of models is usually a little too cautious. Raising
    the probabilities to a power above 1 (and renormalising) sharpens them;
    below 1 flattens them. The power is chosen on out-of-fold predictions and
    kept at 1 unless the gain is clear.
    """
    logs = np.log(np.clip(probs, 1e-9, None))
    rows = np.arange(len(picked))

    def likelihood(power):
        scaled = power * logs
        peak = scaled.max(axis=1, keepdims=True)
        norm = peak[:, 0] + np.log(np.exp(scaled - peak).sum(axis=1))
        return float((scaled[rows, picked] - norm).sum())

    best, best_value = 1.0, likelihood(1.0)
    for power in grid:
        value = likelihood(power)
        if value > best_value + 2.0:
            best, best_value = power, value
    return best


def _sharpen(oof, picked, mixed):
    """Recalibrate one item's mixed predictions, separately for confident
    rows (top probability above POWER_SURE) and for the rest: an average of
    models is most over-cautious exactly where one of them is nearly sure.
    Returns the adjusted held-out predictions, the adjusted out-of-fold
    predictions, and the two powers used."""
    powers = []
    sure_oof = oof.max(axis=1) >= POWER_SURE
    sure_mixed = mixed.max(axis=1) >= POWER_SURE
    oof, mixed = oof.copy(), mixed.copy()
    for sure, grid in ((True, POWER_GRID_SURE), (False, POWER_GRID)):
        rows = sure_oof == sure
        power = 1.0
        if rows.sum() >= POWER_MIN_CELLS:
            power = _best_power(oof[rows], picked[rows], grid)
        powers.append(power)
        if power != 1.0:
            for table, part in ((oof, rows), (mixed, sure_mixed == sure)):
                sharp = np.clip(table[part], 1e-9, None) ** power
                table[part] = sharp / sharp.sum(axis=1, keepdims=True)
    return mixed, oof, powers


# -------------------------------------------------------------------- core --
def _predict_core(frame, schema, deadline, roomy=False):
    share = max(1.0, deadline - time.time())
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
        if flag.any():
            # Trust the gate only where the visible rows bear it out. Some
            # instruments' gates are approximate; there the item is left to
            # the models, which see the parent's answer as an input anyway.
            seen = codes[name][train_rows][flag[train_rows]]
            seen = seen[seen >= 0]
            gate_slot = len(options[name]) - 1
            if seen.size < 20 or np.mean(seen == gate_slot) < 0.98:
                flag = np.zeros(n_rows, dtype=bool)
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

    # ---- sharp count tables with gate recursion --------------------------
    try:
        if given and time.time() < deadline:
            position = {name: j for j, name in enumerate(targets)}
            column = {name: g for g, name in enumerate(given)}
            gates = []
            for name in targets:
                gate = records[name].get("gate")
                parent = gate["parent"] if gate else None
                if not gate:
                    gates.append(None)
                elif parent in column:
                    gates.append(("given", column[parent], [
                        i for i, v in enumerate(options[parent])
                        if v in gate["observed_if"]]))
                elif parent in position:
                    gates.append(("item", position[parent], [
                        i for i, v in enumerate(records[parent]["values"])
                        if v in gate["observed_if"]]))
                else:     # parent not shipped: the gate slot is one more answer
                    gates.append(None)
            for depth in RECURSION_DEPTHS:
                table = _recursion(g_all, y_all, splits, widths, cards, gates,
                                   depth, c0)
                if np.isfinite(table).all():
                    candidates.append(table)
                    names.append("tables(%d items)" % depth)
    except Exception as error:
        _log("tables skipped: %s" % type(error).__name__)

    # ---- gradient-boosted trees, with and without profile matches --------
    if lightgbm is not None and given:
        try:
            x_np = np.zeros((n_rows, int(sum(cards))), dtype=np.float32)
            start = 0
            for g, card in enumerate(cards):
                known = np.flatnonzero(g_all[:, g] >= 0)
                x_np[known, start + g_all[known, g]] = 1.0
                start += card
            twin_item = _twin_item(x_np, g_all, train_rows, cards)
            votes = _profile_votes(x_np, g_all, y_all, train_rows, widths,
                                   twin_item)
            del x_np
            close = np.zeros(train_rows.size, dtype=bool)
            for group in votes.values():
                close |= group["distance"][train_rows] < 0.5
            position = {name: j for j, name in enumerate(targets)}
            parents = [position.get((records[name].get("gate") or {})
                                    .get("parent")) for name in targets]
            _log("twin item: %s; %.0f%% of visible respondents have an "
                 "identical profile match"
                 % (given[twin_item] if twin_item is not None else "none",
                    100.0 * close.mean()))
            workers = int(os.environ.get("SB_WORKERS", "0")) or min(
                8, max(1, os.cpu_count() or 1))
            variants = [("trees+matches", votes), ("trees", None)]
            if close.mean() < 0.05:
                variants.reverse()
            for label, use in variants:
                if time.time() > deadline - 0.15 * share:
                    _log("%s skipped: out of time" % label)
                    continue
                tables, finished = _trees(
                    g_all, y_fit_all, splits, widths, c0, cards, use,
                    parents, deadline, workers)
                if finished < 1.0:
                    _log("%s: %.0f%% of items finished in time"
                         % (label, 100.0 * finished))
                for rounds, table in zip(GBDT_ROUNDS, tables):
                    if np.isfinite(table).all():
                        candidates.append(table)
                        names.append("%s(%d rounds)" % (label, rounds))
        except Exception as error:
            _log("trees skipped: %s" % type(error).__name__)

    # ---- torch candidates, cheapest first, each one optional -------------
    x_all, device = None, "cpu"
    found = {"twin": None}
    if torch is not None and given:
        def usable(table):
            return (np.isfinite(table[train_rows]).all()
                    and np.isfinite(table[hidden_rows]).all())

        try:
            torch.manual_seed(SEED)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            x_np = np.zeros((n_rows, int(sum(cards))), dtype=np.float32)
            start = 0
            for g, card in enumerate(cards):
                known = np.flatnonzero(g_all[:, g] >= 0)
                x_np[known, start + g_all[known, g]] = 1.0
                start += card
            x_all = torch.as_tensor(x_np, device=device)
            if device == "cpu":
                torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
        except Exception as error:
            _log("torch setup failed: %s" % type(error).__name__)

        def run(label, build, cost_guess):
            """Run one candidate family unless the deadline is too close."""
            began = time.time()
            if x_all is None or began + cost_guess > deadline:
                _log("%s skipped: out of time" % label)
                return None
            try:
                built = build()
            except Exception as error:
                _log("%s skipped: %s" % (label, type(error).__name__))
                return None
            return built, time.time() - began

        # Without a GPU everything is several times slower: run a lighter
        # schedule so the whole run stays inside the time limit.
        slow = device == "cpu" and not roomy
        seeds = MLP_SEEDS[:1] if slow else MLP_SEEDS
        l2_strengths = L2_STRENGTHS[1:3] if slow else L2_STRENGTHS
        tilt_l2 = TILT_L2[:1] if slow else TILT_L2

        # c3: neural net, first seed
        mlp_tables, mlp_cost = [], 0.0
        done = run("net seed 0", lambda: _mlp(
            x_all, y_fit_all, splits, widths, c0, device, MLP_SEEDS[0]),
            0.35 * share if slow else 0.0)
        if done:
            mlp_tables.append(done[0])
            mlp_cost = done[1]

        # c2b: logistic regression with the "ordered answers" head
        gated = [bool(records[n].get("gate")) for n in targets]
        for l2 in tilt_l2:
            def tilt(l2=l2):
                basis = _tilt_basis(widths, gated, device)
                table = empty()
                for fit, evaluate in splits:
                    bias = np.log(np.clip(c0[evaluate[0]], 1e-6, None))
                    model = _Linear(widths, device, basis).fit(
                        x_all[fit], y_fit_all[fit], l2, bias.astype(np.float32))
                    table[evaluate] = model.predict(x_all[evaluate])
                return table
            done = run("ordered l2=%g" % l2, tilt, 0.5 * mlp_cost)
            if done and usable(done[0]):
                candidates.append(done[0])
                names.append("ordered(l2=%g)" % l2)

        # c2c: logistic regression with interactions
        try:
            split_on = _interaction_columns(g_all, y_fit_all, train_rows,
                                            cards, widths)
        except Exception:
            split_on = []
        if split_on:
            def interactions(l2):
                starts = np.concatenate([[0], np.cumsum(cards)]).astype(int)
                blocks = [x_all]
                for g in split_on:
                    level = x_all[:, int(starts[g]):int(starts[g + 1])]
                    blocks.append((x_all[:, :, None] * level[:, None, :])
                                  .reshape(n_rows, -1))
                wide = torch.cat(blocks, dim=1)
                table = empty()
                for fit, evaluate in splits:
                    bias = np.log(np.clip(c0[evaluate[0]], 1e-6, None))
                    model = _Linear(widths, device).fit(
                        wide[fit], y_fit_all[fit], l2, bias.astype(np.float32))
                    table[evaluate] = model.predict(wide[evaluate])
                return table
            inter_cost = 0.0
            for l2 in (INTERACTION_L2[:1] if slow else INTERACTION_L2):
                done = run("interactions l2=%g" % l2,
                           lambda l2=l2: interactions(l2),
                           max(mlp_cost, 1.5 * inter_cost))
                if not done:
                    break
                inter_cost = done[1]
                if usable(done[0]):
                    candidates.append(done[0])
                    names.append("interactions(%s; l2=%g)" % (
                        "+".join(given[g] for g in split_on), l2))

        # c4: neural net that also sees what the closest matches answered
        neighbour_tables = []
        def neighbour_features():
            twin = found["twin"] = _twin_column(x_all, train_rows, cards,
                                                 device)
            return [_neighbour_features(x_all, y_fit_all, fit, evaluate,
                                        widths, twin, len(given), device)
                    for fit, evaluate in splits]
        done = run("neighbour features", neighbour_features, 6.0 * mlp_cost)
        if done:
            features = done[0]
            for seed in (NEIGHBOUR_SEEDS[:1] if slow else NEIGHBOUR_SEEDS):
                done = run("neighbour net seed %d" % seed,
                           lambda seed=seed: _mlp(
                               x_all, y_fit_all, splits, widths, c0, device,
                               100 + seed, features), 4.0 * mlp_cost)
                if not done:
                    break
                neighbour_tables.append(done[0])
            del features
        if neighbour_tables:
            for index, epochs in enumerate(MLP_CHECKPOINTS):
                table = (sum(t[index] for t in neighbour_tables)
                         / len(neighbour_tables))
                if usable(table):
                    candidates.append(table)
                    names.append("neighbour net(%d epochs, %d seeds)"
                                 % (epochs, len(neighbour_tables)))

        # c2: logistic regression, strongest penalty first
        last_cost = 0.0
        for l2 in l2_strengths:
            def logistic(l2=l2):
                table = empty()
                for fit, evaluate in splits:
                    bias = np.log(np.clip(c0[evaluate[0]], 1e-6, None))
                    model = _Linear(widths, device).fit(
                        x_all[fit], y_fit_all[fit], l2, bias.astype(np.float32))
                    table[evaluate] = model.predict(x_all[evaluate])
                return table
            done = run("logistic l2=%g" % l2, logistic, 1.5 * last_cost)
            if not done:
                break
            last_cost = done[1]
            if usable(done[0]):
                candidates.append(done[0])
                names.append("logistic(l2=%g)" % l2)

        # c3: further seeds, averaged with the first
        for seed in seeds[1:]:
            if not mlp_tables:
                break
            done = run("net seed %d" % seed, lambda seed=seed: _mlp(
                x_all, y_fit_all, splits, widths, c0, device, seed),
                1.3 * mlp_cost)
            if not done:
                break
            mlp_tables.append(done[0])
        if mlp_tables:
            for index, epochs in enumerate(MLP_CHECKPOINTS):
                table = sum(t[index] for t in mlp_tables) / len(mlp_tables)
                if usable(table):
                    candidates.append(table)
                    names.append("net(%d epochs, %d seeds)"
                                 % (epochs, len(mlp_tables)))

    # ---- mix per item on the out-of-fold log loss ------------------------
    def cells_of(j):
        y = y_fit_all[train_rows, j]
        return train_rows[y >= 0], y[y >= 0]

    smooth = empty()
    usage = np.zeros(len(candidates))
    for j in range(len(targets)):
        lo, hi = offsets[j], offsets[j + 1]
        rows, picked = cells_of(j)
        weights = _mixture_weights(np.stack(
            [c[rows, lo + picked] for c in candidates], axis=1))
        usage += weights
        smooth[:, lo:hi] = sum(w * c[:, lo:hi]
                               for w, c in zip(weights, candidates))
    _log("mean weight: " + ", ".join(
        "%s %.2f" % (n, u / len(targets)) for n, u in zip(names, usage)))

    # ---- profile matching on top of the mixed model ----------------------
    final = [smooth]
    if x_all is not None and time.time() < deadline + 60:
        try:
            matched = _match(x_all, y_fit_all, splits, widths, smooth,
                             len(given), device)
            drop = _best_drop(x_all, y_fit_all, splits[:-1], widths, smooth,
                              cards, device, train_rows, deadline + 60)
            for drop in dict.fromkeys([drop, found["twin"]]):
                if drop is None:
                    continue
                _log("profile matching also ignores GIVEN item %s"
                     % given[int(np.searchsorted(np.cumsum(cards), drop[0],
                                                 side="right"))])
                matched += _match(x_all, y_fit_all, splits, widths, smooth,
                                  len(given), device, drop=drop,
                                  reaches=MATCH_DROP_REACH)
            final += [t for t in matched
                      if np.isfinite(t[train_rows]).all()
                      and np.isfinite(t[hidden_rows]).all()]
        except Exception as error:
            _log("profile matching skipped: %s" % type(error).__name__)

    result = {}
    powers = np.ones((len(targets), 2))
    usage = np.zeros(len(final))
    total_mix, total_smooth, total_base, total_cells = 0.0, 0.0, 0.0, 0
    for j, name in enumerate(targets):
        lo, hi = offsets[j], offsets[j + 1]
        rows, picked = cells_of(j)
        true_probs = np.stack([c[rows, lo + picked] for c in final], axis=1)
        weights = _mixture_weights(true_probs)
        usage += weights
        mixed = sum(w * c[hidden_rows, lo:hi] for w, c in zip(weights, final))
        mixed_true = true_probs @ weights
        if rows.size >= POWER_MIN_CELLS:
            oof = sum(w * c[rows, lo:hi] for w, c in zip(weights, final))
            mixed, oof, used = _sharpen(oof, picked, mixed)
            powers[j] = used
            mixed_true = oof[np.arange(rows.size), picked]

        # Rows the skip logic decides: that stratum's own distribution, with
        # a prior on the gate slot (the last one) in case it is empty.
        stuck = decided_all[hidden_rows, j]
        if stuck.any():
            seen = y_all[train_rows, j][decided_all[train_rows, j]]
            extra = np.zeros(widths[j])
            extra[-1] = 5.0
            counts = np.bincount(seen[seen >= 0], minlength=widths[j])
            if counts.sum() >= 30:
                # Plenty of evidence: keep only a trace of smoothing.
                mixed[stuck] = (counts + 0.02 + extra) / (
                    counts + 0.02 + extra).sum()
            else:
                mixed[stuck] = _smoothed(counts, extra)
        result[name] = mixed

        if rows.size:
            clip = lambda v: np.log(np.clip(v, 1e-12, None)).sum()
            total_mix += clip(mixed_true)
            total_smooth += clip(true_probs[:, 0])
            total_base += clip(c0[rows, lo + picked])
            total_cells += rows.size

    if total_cells and total_base < 0:
        _log("%d items, %d visible rows; out-of-fold skill on undecided "
             "cells: models %.4f, with profile matching %.4f; matching "
             "weight by reach: %s"
             % (len(targets), train_rows.size,
                1.0 - total_smooth / total_base, 1.0 - total_mix / total_base,
                ", ".join("%.2f" % (u / len(targets)) for u in usage[1:])))
        _log("sharpening: mean power %.2f on confident rows, %.2f elsewhere"
             % (powers[:, 0].mean(), powers[:, 1].mean()))

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
    budget = TOTAL_BUDGET
    if torch is None or not torch.cuda.is_available():
        budget -= 100.0          # no GPU: leave a wider safety margin
    try:
        visible = int((~frame.isna().any(axis=1)).sum())
        if visible > int(schema["split"]["n_train"]):
            budget = max(budget, FINAL_BUDGET)
    except Exception:
        pass
    remaining = budget - (time.time() - _T0)
    share = remaining / max(1, N_INSTRUMENTS - _calls + 1)
    deadline = time.time() + max(0.0, share)
    try:
        return _predict_core(frame, schema, deadline, budget >= FINAL_BUDGET)
    except Exception as error:
        _log("falling back to the crowd marginal: %s" % type(error).__name__)
        return _marginal_predict(frame, schema)
