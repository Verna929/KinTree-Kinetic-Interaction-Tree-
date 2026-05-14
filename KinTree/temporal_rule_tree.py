from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Sequence, Iterable, Union
import numpy as np
import json
from pathlib import Path
import argparse, torch, pickle
from scipy.stats import rankdata

# -------------------------
# Utilities (metrics)
# -------------------------

def _onehot_binary(z: np.ndarray) -> np.ndarray:
    """One-hot encoding for a binary vector (values in {0,1}). Returns (N,2)."""
    z = (z > 0).astype(np.int8)
    oh = np.zeros((z.size, 2), dtype=float)
    oh[np.arange(z.size), z] = 1.0
    return oh


def _contingency_2x2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return 2×2 contingency table for binary x,y in {0,1}."""
    x = (x > 0).astype(np.int8)
    y = (y > 0).astype(np.int8)
    C = np.zeros((2, 2), dtype=float)
    for a in (0, 1):
        for b in (0, 1):
            C[a, b] = np.sum((x == a) & (y == b))
    return C


def _nmi_binary(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized mutual information for binary variables x,y ∈ {0,1}.
    NMI = I(X;Y) / sqrt(H(X)H(Y)) in [0,1].
    """
    C = _contingency_2x2(x, y) + eps
    P = C / C.sum()
    px = P.sum(axis=1, keepdims=True)
    py = P.sum(axis=0, keepdims=True)
    I = np.sum(P * np.log(P / (px @ py)))
    Hx = -np.sum(px * np.log(px))
    Hy = -np.sum(py * np.log(py))
    denom = np.sqrt(max(Hx, eps) * max(Hy, eps))
    return float(I / denom) if denom > 0 else 0.0


def _vamp2_binary(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> float:
    """VAMP-2 score for a binary split variable at t predicting itself at t+τ.
    Uses centered one-hot features; returns squared Frobenius norm of Koopman estimate.
    """
    X = _onehot_binary(x)
    Y = _onehot_binary(y)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    n = X.shape[0] if X.shape[0] > 0 else 1
    C00 = (Xc.T @ Xc) / n
    Ctt = (Yc.T @ Yc) / n
    C0t = (Xc.T @ Yc) / n
    C00 = C00 + ridge * np.eye(C00.shape[0])
    Ctt = Ctt + ridge * np.eye(Ctt.shape[0])

    def inv_sqrt(M: np.ndarray) -> np.ndarray:
        w, V = np.linalg.eigh(M)
        w = np.clip(w, 1e-12, None)
        return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

    W0 = inv_sqrt(C00)
    Wt = inv_sqrt(Ctt)
    K = W0 @ C0t @ Wt
    return float(np.sum(K * K))


# -------------------------
# Temporal persistence / hysteresis
# -------------------------
def _hysteresis_debounce_1d(x: np.ndarray, min_on: int = 3, min_off: int = 3) -> np.ndarray:
    """
    Debounce a single binary time series x (shape [T], values {0,1}).
    - To switch 0 -> 1, we require >= min_on consecutive 1s.
    - To switch 1 -> 0, we require >= min_off consecutive 0s.
    Returns a new array y with the same shape, debounced.
    """
    x = (x > 0).astype(np.int8)
    T = x.shape[0]
    y = np.zeros(T, dtype=np.int8)

    state = 0
    streak = 0

    for t in range(T):
        if state == 0:
            if x[t] == 1:
                streak += 1
                if streak >= min_on:
                    state = 1
                    y[t - min_on + 1 : t + 1] = 1
                    streak = 0
            else:
                streak = 0
            if t > 0 and y[t] == 0:
                y[t] = y[t - 1]
        else:  # state == 1
            if x[t] == 0:
                streak += 1
                if streak >= min_off:
                    state = 0
                    streak = 0
            else:
                streak = 0
            y[t] = 1 if state == 1 else 0
    return y


def apply_hysteresis(
        X: np.ndarray,
        min_on: int = 3,
        min_off: int = 3,
        traj_lengths: Optional[Sequence[int]] = None  # <-- ADD THIS
) -> np.ndarray:
    """
    Apply hysteresis to every feature column independently.
    If traj_lengths is provided, applies debouncing to each trajectory
    segment separately to avoid cross-boundary artifacts.
    """
    X = (X > 0).astype(np.int8)
    T, F = X.shape
    Y = np.empty_like(X)

    if traj_lengths is None:
        print("Applying hysteresis to all features (single trajectory)...")
        for f in range(F):
            Y[:, f] = _hysteresis_debounce_1d(X[:, f], min_on=min_on, min_off=min_off)
    else:
        print(f"Applying hysteresis to all features ({len(traj_lengths)} segments)...")
        if sum(traj_lengths) != T:
            raise ValueError(
                f"Sum of traj_lengths ({sum(traj_lengths)}) "
                f"does not match X.shape[0] ({T})"
            )

        start_idx = 0
        for length in traj_lengths:
            end_idx = start_idx + length
            segment_X = X[start_idx:end_idx, :]
            segment_Y = np.empty_like(segment_X)

            for f in range(F):
                segment_Y[:, f] = _hysteresis_debounce_1d(
                    segment_X[:, f], min_on=min_on, min_off=min_off
                )

            Y[start_idx:end_idx, :] = segment_Y
            start_idx = end_idx

    return Y

# =========================
# Sanity Check for Hysteresis Filtering
# =========================
def _binary_runs(seq):
    """
    Return list of runs:
    (value, start, end, length)
    end is exclusive
    """
    seq = np.asarray(seq).astype(int)
    runs = []
    if len(seq) == 0:
        return runs

    s = 0
    cur = seq[0]
    for i in range(1, len(seq)):
        if seq[i] != cur:
            runs.append((cur, s, i, i - s))
            s = i
            cur = seq[i]
    runs.append((cur, s, len(seq), len(seq) - s))
    return runs


def _collect_run_stats(raw, smooth, short_on_thr=3, short_off_thr=3):
    """
    Summarize how smoothing changed short vs long raw runs.
    """
    raw = np.asarray(raw).astype(int)
    smooth = np.asarray(smooth).astype(int)
    edited = (raw != smooth)

    runs = _binary_runs(raw)

    short_off_filled_frames = 0
    short_on_removed_frames = 0
    long_run_edited_frames = 0

    n_short_off_runs = 0
    n_short_off_runs_filled = 0
    n_short_on_runs = 0
    n_short_on_runs_removed = 0

    for val, s, e, L in runs:
        raw_seg = raw[s:e]
        smooth_seg = smooth[s:e]
        edited_seg = edited[s:e]

        if val == 0:
            if L <= short_off_thr:
                n_short_off_runs += 1
                if np.mean(smooth_seg == 1) > 0.5:
                    n_short_off_runs_filled += 1
                    short_off_filled_frames += L
            else:
                long_run_edited_frames += edited_seg.sum()

        else:  # val == 1
            if L <= short_on_thr:
                n_short_on_runs += 1
                if np.mean(smooth_seg == 0) > 0.5:
                    n_short_on_runs_removed += 1
                    short_on_removed_frames += L
            else:
                long_run_edited_frames += edited_seg.sum()

    return {
        "short_off_filled_frames": int(short_off_filled_frames),
        "short_on_removed_frames": int(short_on_removed_frames),
        "long_run_edited_frames": int(long_run_edited_frames),
        "n_short_off_runs": int(n_short_off_runs),
        "n_short_off_runs_filled": int(n_short_off_runs_filled),
        "n_short_on_runs": int(n_short_on_runs),
        "n_short_on_runs_removed": int(n_short_on_runs_removed),
    }


def make_partial_revert_mask(raw, smooth, short_on_thr=3, short_off_thr=3):
    """
    Return a boolean mask of positions to revert back to raw
    for an ambiguous feature.

    Rule:
    - revert edited positions that lie on LONG raw runs
    - keep edited positions that lie on SHORT raw runs

    This preserves denoising of flicker while undoing edits that
    reshape longer dwell periods.
    """
    raw = np.asarray(raw).astype(int)
    smooth = np.asarray(smooth).astype(int)

    edited = (raw != smooth)
    revert_mask = np.zeros_like(edited, dtype=bool)

    runs = _binary_runs(raw)

    for val, s, e, L in runs:
        if val == 0:
            is_long = L > short_off_thr
        else:
            is_long = L > short_on_thr

        if is_long:
            revert_mask[s:e] |= edited[s:e]

    return revert_mask


def classify_feature_action(
    raw,
    smooth,
    T,
    short_on_thr=3,
    short_off_thr=3,
    pct_modified_thr=0.15,
    min_raw_support=0.02,
    max_occupancy_shift=0.15,
    safe_occupancy_shift=0.05,
    safe_helpful_ratio=0.70,
    max_long_run_damage=0.20,
    common_feature_raw_frac=0.10,
):
    """
    Classify a feature into:
    - keep_smoothed
    - partial_revert
    - full_revert
    """
    raw = np.asarray(raw).astype(int)
    smooth = np.asarray(smooth).astype(int)

    n_raw = int((raw == 1).sum())
    n_smooth = int((smooth == 1).sum())

    n_added = int(((raw == 0) & (smooth == 1)).sum())
    n_removed = int(((raw == 1) & (smooth == 0)).sum())
    n_total = n_added + n_removed
    pct_modified = n_total / T

    raw_frac = n_raw / T
    smooth_frac = n_smooth / T
    occupancy_shift = abs(smooth_frac - raw_frac)

    run_stats = _collect_run_stats(
        raw, smooth,
        short_on_thr=short_on_thr,
        short_off_thr=short_off_thr
    )

    helpful_edit_frames = (
        run_stats["short_off_filled_frames"] +
        run_stats["short_on_removed_frames"]
    )
    helpful_edit_ratio = helpful_edit_frames / max(n_total, 1)
    long_run_damage = run_stats["long_run_edited_frames"] / max(n_total, 1)

    # default
    action = "keep_smoothed"
    reason = "below_pct_threshold"

    # 1. low modification -> keep
    if pct_modified <= pct_modified_thr:
        action = "keep_smoothed"
        reason = "below_pct_threshold"

    else:
        # 2. clearly artificial / distorted -> full revert
        if raw_frac < min_raw_support and smooth_frac > raw_frac * 2:
            action = "full_revert"
            reason = "rare_feature_artificially_inflated"

        elif occupancy_shift > max_occupancy_shift:
            action = "full_revert"
            reason = "large_occupancy_shift"

        # 3. clearly safe cleanup -> keep
        elif helpful_edit_ratio >= safe_helpful_ratio and long_run_damage <= max_long_run_damage:
            action = "keep_smoothed"
            reason = "mostly_short_run_cleanup"

        # 4. common feature, overall occupancy stable, but mixed behavior -> partial revert
        elif raw_frac >= common_feature_raw_frac and occupancy_shift <= safe_occupancy_shift:
            action = "partial_revert"
            reason = "common_feature_small_occupancy_shift"

        # 5. mixed / suspicious -> partial revert
        elif long_run_damage > max_long_run_damage:
            action = "partial_revert"
            reason = "mixed_feature_long_run_damage"

        # 6. default ambiguous -> partial revert
        else:
            action = "partial_revert"
            reason = "ambiguous_partial_revert"

    out = {
        "raw_ON": n_raw,
        "smooth_ON": n_smooth,
        "added_fake_ON": n_added,
        "removed_true_ON": n_removed,
        "total_modified": n_total,
        "pct_modified": pct_modified,
        "raw_frac": raw_frac,
        "smooth_frac": smooth_frac,
        "occupancy_shift": occupancy_shift,
        "helpful_edit_ratio": helpful_edit_ratio,
        "long_run_damage": long_run_damage,
        "action_new": action,
        "reason_new": reason,
    }
    out.update(run_stats)
    return out


# -------------------------
# Pair handling (time lag)
# -------------------------

def make_pairs(T: int, tau: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Return index pairs (t, t+tau) for 0 ≤ t < T-τ. If mask is given (bool length T),
    keep only pairs where both endpoints are True.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    t0 = np.arange(0, max(0, T - tau), dtype=int)
    t1 = t0 + tau
    if mask is not None:
        keep = mask[t0] & mask[t1]
        t0 = t0[keep]
        t1 = t1[keep]
    return np.stack([t0, t1], axis=1) if t0.size else np.zeros((0, 2), dtype=int)


# -------------------------
# Data loaders
# -------------------------

def load_interaction_binary_pt(
    pt_path: str, use_types_separately: bool = True, present_threshold: float = 0.5
):
    data = torch.load(pt_path, map_location="cpu")
    edge_index = np.array(data["edge_index"])
    edge_attr_all = data["edge_attr_all"]
    itypes = list(data["interaction_types"])
    E = edge_index.shape[1]
    K = len(itypes)

    edge_names = [f"{int(edge_index[0,e])}-{int(edge_index[1,e])}" for e in range(E)]
    if use_types_separately:
        feature_names = [f"{edge_names[e]}:{itypes[t]}" for e in range(E) for t in range(K)]
        feature_types = [itypes[t] for e in range(E) for t in range(K)]
        F = E * K
    else:
        feature_names = [f"{edge_names[e]}:any" for e in range(E)]
        feature_types = ["any"] * E
        F = E

    T = len(edge_attr_all)
    X = np.zeros((T, F), dtype=np.int8)
    for t, attr in enumerate(edge_attr_all):
        A = attr.detach().cpu().numpy()
        if use_types_separately:
            B = (A >= present_threshold).astype(np.int8)
            X[t] = B.reshape(-1)
        else:
            B = (A >= present_threshold).any(axis=1).astype(np.int8)
            X[t] = B
    return X, feature_names, feature_types


def load_contact_map_pkl(pkl_path: str) -> List[List[Tuple[int, int, float]]]:
    """Load soft contact map frames from a pickle file.
    Expects a list over frames; each frame is a list of (i, j, score) with 0<=score<=1.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError("contact_pkl must store a list over frames; each frame is a list of (i,j,score)")
    return data

# -------------------------
# pre-processing and post-processing
# -------------------------

def seqsep_soft_weights(feature_names, min_sep=5, gamma=2.0, floor=0.2, by_type=None):
    """
    Weight formula (sep <= min_sep):  w = max(floor, (sep / min_sep)**gamma)
    Otherwise: w = 1.
    """
    w = np.ones(len(feature_names), float)
    for k, name in enumerate(feature_names):
        pair, _, itype = name.partition(":")
        i, j = map(int, pair.split("-"))
        sep_thr = by_type.get(itype, min_sep) if by_type else min_sep
        sep = abs(i - j)
        if sep <= sep_thr:
            w[k] = max(floor, (sep / sep_thr) ** gamma)
    return w

def save_tree_json(model, path: str):
    """
    Export the tree structure from a fitted model and save as JSON.
    """
    tree_dict = model.export_tree()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(tree_dict, f, indent=2)
    print(f"Tree JSON saved to {path}")
    return tree_dict

def collect_split_features(node, model, feats=None):
    if feats is None: feats = []
    if node.get("leaf", False): return feats
    rule = node["rule"]
    feats.append(rule["name"])
    collect_split_features(node["left"], model, feats)
    collect_split_features(node["right"], model, feats)
    return feats

# -------------------------
# Tree structures
# -------------------------

@dataclass
class Rule:
    feature: int
    polarity: int = 1  # 1 means "present" branch is left, 0 means inverted

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask for going to LEFT child (True = go left)."""
        f = X[:, self.feature] > 0
        return f if self.polarity == 1 else ~f


@dataclass
class Node:
    idx: np.ndarray
    depth: int
    rule: Optional[Rule] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    leaf_id: Optional[int] = None
    score_here: float = 0.0
    n_pairs: int = 0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# -------------------------
# Main model
# -------------------------

class TemporalRuleTreeV6:
    def __init__(
        self,
        lags: Sequence[int],
        alpha_vamp: float = 0.7,  # weight for VAMP-2 vs NMI
        ridge: float = 1e-6,
        min_pairs: int = 200,
        min_leaf: int = 100,
        max_depth: int = 8,
        max_features_to_try: Optional[int] = None,
        feature_types: Optional[Sequence[str]] = None,  # same length as F
        stratified_subsample: bool = True,
        random_state: Optional[int] = 0,
        lag_mode: str = "uniform",  # "uniform" | "max" | "power"
        tau_power: float = 1.0,      # used when lag_mode=="power"
        lambda_row: float = 0.25,    # penalty weight for child-row similarity
        contact_frames: Optional[List[List[Tuple[int,int,float]]]] = None,
        local_window: int = 2,       # exclude contacts near the split pair (sequence window)
        delta_mode: str = "quantile",   # "quantile" | "threshold"
        delta_quantile: float = 0.6,     # used when delta_mode=="quantile"
        delta_thresh: float = 0.2,       # used when delta_mode=="threshold" (on summed change)
        contact_weight_spec: Optional[Dict[str, Any]] = None,  # e.g., {"active_site_residues":[..], "weight":3.0}
        res_chain_ids: Optional[Sequence[int]] = None,  # chain id per residue index
    ):
        self.lags = list(sorted(set(int(t) for t in lags)))
        self.alpha = float(alpha_vamp)
        self.ridge = float(ridge)
        self.min_pairs = int(min_pairs)
        self.min_leaf = int(min_leaf)
        self.max_depth = int(max_depth)
        self.max_features_to_try = max_features_to_try
        self.feature_types = list(feature_types) if feature_types is not None else None
        self.stratified = bool(stratified_subsample)
        self.rng = np.random.default_rng(random_state)
        self._feat_w: Optional[np.ndarray] = None
        self._type_w_map: Optional[Dict[str, float]] = None
        self._w: Optional[np.ndarray] = None
        self.root_: Optional[Node] = None
        self.leaf_indices_: List[np.ndarray] = []
        self.leaf_ids_: Optional[np.ndarray] = None
        self.pairs_by_lag_: Dict[int, np.ndarray] = {}
        self.feature_names_: Optional[List[str]] = None
        self.lag_mode = str(lag_mode)
        if self.lag_mode not in {"uniform", "max", "power"}:
            raise ValueError("lag_mode must be one of {'uniform','max','power'}")
        self.tau_power = float(tau_power)
        self.lambda_row = float(lambda_row)
        self.contact_frames_: Optional[List[Dict[Tuple[int,int], float]]] = None
        self.local_window = int(local_window)
        self.delta_mode = str(delta_mode)
        self.delta_quantile = float(delta_quantile)
        self.delta_thresh = float(delta_thresh)
        self.contact_weight_spec = contact_weight_spec
        self.feature_pairs_: Optional[List[Tuple[int,int]]] = None
        self.res_chain_ids = None if res_chain_ids is None else np.asarray(res_chain_ids, dtype=np.int32)
        if contact_frames is not None:
            self._ingest_contact_frames(contact_frames)
        self._deltaC_cache_: Dict[int, np.ndarray] = {}
        if contact_frames is not None:
            self._precompute_deltaC()
        self.depth_debug = 0

    # ---------- contact helpers ----------
    def _ingest_contact_frames(self, frames: List[List[Tuple[int,int,float]]]):
        """Convert list-of-lists into list of dicts {(i,j):score} for O(1) lookup."""
        dicts = []
        for frame in frames:
            d: Dict[Tuple[int,int], float] = {}
            for i, j, s in frame:
                key = (int(i), int(j)) if i <= j else (int(j), int(i))
                d[key] = float(s)
            dicts.append(d)
        self.contact_frames_ = dicts
        self._contact_weight = self._build_contact_weight_fn(self.contact_weight_spec)

    def _precompute_deltaC(self):
        """Precompute summed |ΔC| per lag for all frame pairs, store as vector per τ."""
        T = len(self.contact_frames_)
        for tau in self.lags:
            vals = np.zeros(max(0, T - tau), dtype=float)
            for t in range(0, T - tau):
                vals[t] = self._delta_contact_sum(t, t + tau, (-1, -1))
            self._deltaC_cache_[tau] = vals

    def _build_contact_weight_fn(self, spec: Optional[Dict[str, Any]]):
        """
        Build a fast NumPy-based weight lookup function.
        Avoids Python 'in' checks per contact by precomputing an array of per-residue weights.
        """
        if spec is None:
            return lambda p, q: 1.0

        residues = spec.get("active_site_residues", [])
        weight = float(spec.get("weight", 1.0))
        if not residues or weight == 1.0:
            return lambda p, q: 1.0

        # --- Precompute lookup array ---
        max_resid = max(residues) + 1
        res_weight = np.ones(max_resid, dtype=np.float32)
        res_weight[residues] = weight

        def weight_fn(p: int, q: int) -> float:
            if p >= res_weight.size or q >= res_weight.size:
                return 1.0
            wp = res_weight[p]
            wq = res_weight[q]
            return wp if wp >= wq else wq

        return weight_fn

    @staticmethod
    def _parse_pair_from_name(name: str) -> Tuple[int,int]:
        # expects "i-j:type" or "i-j:any"
        pair = name.split(":", 1)[0]
        i_str, j_str = pair.split("-")
        return int(i_str), int(j_str)

    def _delta_contact_sum(self, t0: int, t1: int, split_pair: Tuple[int, int]) -> float:
        """
        Compute the mean weighted |C(t1)-C(t0)| over all contacts,
        excluding local neighborhoods near the split residues.

        Chain-aware: local masking only applies when a contact residue and the split residue
        are in the same chain (if res_chain_ids is provided).

        Also fixes a bug: if split_pair has invalid indices (e.g. (-1,-1) during precompute),
        local masking is disabled (previously it could mask residues near index 0/1).
        """
        if self.contact_frames_ is None:
            return 0.0

        d0 = self.contact_frames_[t0]
        d1 = self.contact_frames_[t1]
        i, j = split_pair
        wloc = self.local_window

        use_local_mask = (i is not None and j is not None and i >= 0 and j >= 0)

        def is_local(p: int, ref: int) -> bool:
            if not use_local_mask:
                return False
            if self.res_chain_ids is None:
                return abs(p - ref) <= wloc
            if p < 0 or ref < 0:
                return False
            if p >= self.res_chain_ids.size or ref >= self.res_chain_ids.size:
                return False
            return (self.res_chain_ids[p] == self.res_chain_ids[ref]) and (abs(p - ref) <= wloc)

        total = 0.0
        count = 0
        keys: Iterable[Tuple[int, int]] = set(d0.keys()) | set(d1.keys())

        for (p, q) in keys:
            if is_local(p, i) or is_local(q, i) or is_local(p, j) or is_local(q, j):
                continue

            s0 = d0.get((p, q), 0.0)
            s1 = d1.get((p, q), 0.0)
            if s0 == 0.0 and s1 == 0.0:
                continue

            w_contact = self._contact_weight(p, q) if hasattr(self, "_contact_weight") else 1.0
            total += w_contact * abs(s1 - s0)
            count += 1

        return 0.0 if count == 0 else (total / count)

    @staticmethod
    def make_pairs_with_boundaries(T: int, lags: Sequence[int], traj_lengths: Sequence[int]) -> Dict[int, np.ndarray]:
        """
        Generate valid (t, t+tau) pairs for multi-trajectory data.
        Prevents transitions across trajectory boundaries.

        Parameters
        ----------
        T : int
            Total number of frames (sum of traj_lengths)
        lags : list[int]
            List of lag times
        traj_lengths : list[int]
            Length of each trajectory in order

        Returns
        -------
        dict[int, np.ndarray]
            Mapping tau → (N_tau, 2) index array
        """
        mask = np.ones(T, dtype=bool)
        offset = 0
        for L in traj_lengths[:-1]:
            offset += L
            mask[offset] = False
        return {tau: make_pairs(T, tau, mask=mask) for tau in lags}

    # ---------- weights ----------
    def set_weights(
        self,
        feature_weights: Optional[Sequence[float]] = None,
        type_weight_map: Optional[Dict[str, float]] = None,
    ) -> "TemporalRuleTreeV6":
        """Optionally set per-feature and/or per-type weights."""
        if feature_weights is not None:
            self._feat_w = np.asarray(feature_weights, dtype=float)
        if type_weight_map is not None:
            self._type_w_map = dict(type_weight_map)
        return self

    # ---------- public API ----------
    def fit(self,
            X: np.ndarray,
            feature_names: Optional[Sequence[str]] = None,
            traj_lengths: Optional[Sequence[int]] = None  # <-- ADD THIS ARGUMENT
            ) -> "TemporalRuleTreeV6":
        X = (X > 0).astype(np.int8)
        T, F = X.shape
        self.feature_names_ = (
            list(feature_names) if feature_names is not None else [f"feat_{i}" for i in range(F)]
        )
        self.feature_pairs_ = [self._parse_pair_from_name(nm) for nm in self.feature_names_]

        w = np.ones(F, dtype=float)

        if self._feat_w is not None:
            if self._feat_w.shape[0] != F:
                raise ValueError(f"feature_weights length {self._feat_w.shape[0]} != F={F}")
            w *= self._feat_w

        if self._type_w_map is not None and self.feature_types is not None:
            if len(self.feature_types) != F:
                raise ValueError(
                    "feature_types length must equal number of features when using type_weight_map"
                )

            for i, t in enumerate(self.feature_types):
                if t in self._type_w_map:
                    w[i] *= float(self._type_w_map[t])
        self._w = w

        if traj_lengths is None:
            print(
                "WARNING: traj_lengths not provided to fit(). "
                "Assuming a single continuous trajectory."
            )
            self.pairs_by_lag_ = {tau: make_pairs(T, tau) for tau in self.lags}
        else:
            if sum(traj_lengths) != T:
                raise ValueError(
                    f"Sum of traj_lengths ({sum(traj_lengths)}) "
                    f"does not match X.shape[0] ({T})"
                )
            print(f"Generating boundary-aware pairs for {len(traj_lengths)} trajectories.")
            self.pairs_by_lag_ = self.make_pairs_with_boundaries(T, self.lags, traj_lengths)

        if self.contact_frames_ is not None and len(self.contact_frames_) != T:
            raise ValueError(f"contact_pkl frames T={len(self.contact_frames_)} must match X frames T={T}")

        all_idx = np.arange(T, dtype=int)
        self.root_ = Node(idx=all_idx, depth=0)
        self._split_recursive(self.root_, X)
        self._finalize_leaves(X)
        return self

    def predict_leaf_ids(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("Model not fitted.")
        X = (X > 0).astype(np.int8)
        T = X.shape[0]
        out = np.empty(T, dtype=int)
        for i in range(T):
            node = self.root_
            while not node.is_leaf():
                go_left = node.rule.apply(X[i : i + 1, :])[0]
                node = node.left if go_left else node.right
            out[i] = node.leaf_id
        return out

    def export_tree(self) -> Dict[str, Any]:
        """Export a nested dict for serialization/plotting."""
        def rec(n: Node) -> Dict[str, Any]:
            d: Dict[str, Any] = {
                "depth": n.depth,
                "n_frames": int(n.idx.size),
                "n_pairs": int(n.n_pairs),
                "leaf": n.is_leaf(),
            }
            if n.is_leaf():
                d["leaf_id"] = int(n.leaf_id)
                return d
            assert n.rule is not None
            feat_name = self.feature_names_[n.rule.feature]
            d["rule"] = {
                "feature": int(n.rule.feature),
                "name": feat_name,
                "polarity_left_is_present": int(n.rule.polarity),
            }
            d["left"] = rec(n.left)
            d["right"] = rec(n.right)
            return d

        return rec(self.root_)

    def occupancy(self) -> np.ndarray:
        if self.leaf_ids_ is None:
            raise RuntimeError("Model not fitted.")
        L = int(self.leaf_ids_.max()) + 1
        counts = np.bincount(self.leaf_ids_, minlength=L).astype(float)
        return counts / counts.sum()

    def _split_recursive(self, node: Node, X: np.ndarray):
        if node.depth >= self.max_depth:
            return
        if node.idx.size < 2 * self.min_leaf:
            return
        mask_node = np.zeros(X.shape[0], dtype=bool)
        mask_node[node.idx] = True
        pairs_per_lag: Dict[int, np.ndarray] = {}
        total_pairs = 0
        for tau, PP in self.pairs_by_lag_.items():
            if PP.size == 0:
                pairs_per_lag[tau] = PP
                continue
            keep = mask_node[PP[:, 0]] & mask_node[PP[:, 1]]
            Pn = PP[keep]
            pairs_per_lag[tau] = Pn
            total_pairs += Pn.shape[0]
        node.n_pairs = int(total_pairs)
        if total_pairs < self.min_pairs:
            return
        Xi = X[node.idx, :]
        varies = (Xi.max(axis=0) != Xi.min(axis=0))
        cand = np.where(varies)[0]
        if cand.size == 0:
            return
        cand = self._subsample_features(cand)

        valid_rules = []
        valid_left_idx = []
        valid_right_idx = []
        raw_vamps = []
        raw_nmis = []
        raw_sims = []

        for f in cand:
            rule = Rule(feature=int(f), polarity=1)
            left_mask = rule.apply(X)
            left_idx = node.idx[left_mask[node.idx]]
            right_idx = node.idx[~left_mask[node.idx]]

            if left_idx.size < self.min_leaf or right_idx.size < self.min_leaf:
                continue

            v2, nmi, sim = self._split_metrics(
                X, left_mask, pairs_per_lag, split_pair=self.feature_pairs_[f]
            )

            valid_rules.append(rule)
            valid_left_idx.append(left_idx)
            valid_right_idx.append(right_idx)
            raw_vamps.append(v2)
            raw_nmis.append(nmi)
            raw_sims.append(sim)

        m = len(valid_rules)
        if m == 0:
            # No valid splits found
            return

        v2_pct = rankdata(raw_vamps, method="average") / m
        nmi_pct = rankdata(raw_nmis, method="average") / m
        sim_pct = rankdata(raw_sims, method="average") / m

        scores = (
                self.alpha * v2_pct +
                (1.0 - self.alpha) * nmi_pct -
                self.lambda_row * sim_pct
        )

        valid_features = np.array([r.feature for r in valid_rules], dtype=int)
        feature_weights = self._feature_weight(valid_features)
        weighted_scores = scores * feature_weights

        best_idx = int(np.argmax(weighted_scores))
        best_score = weighted_scores[best_idx]

        if best_score <= 0:
            # Best split is not better than not splitting
            return

        best_rule = valid_rules[best_idx]
        node.rule = best_rule
        node.left = Node(idx=valid_left_idx[best_idx], depth=node.depth + 1)
        node.right = Node(idx=valid_right_idx[best_idx], depth=node.depth + 1)

        best_f = best_rule.feature
        raw_score_unweighted = scores[best_idx]
        fw = feature_weights[best_idx]

        print(
            f"[RankedGain D={node.depth}] {self.feature_names_[best_f]} (VAMP={raw_vamps[best_idx]:.3f}, NMI={raw_nmis[best_idx]:.3f}) | "
            f"RankScore={raw_score_unweighted:.4f} * w={fw:.2f} = Final={best_score:.4f} | "
            f"Best of {m} candidates.")

        if self.depth_debug == node.depth:
            print(f"  [Debug D={node.depth}] best={self.feature_names_[best_f]} | VAMP_raw={raw_vamps[best_idx]:.3f} "
                  f"NMI_raw={raw_nmis[best_idx]:.3e} sim_raw={raw_sims[best_idx]:.3f} | "
                  f"VAMP_pct={v2_pct[best_idx]:.3f} NMI_pct={nmi_pct[best_idx]:.3f} sim_pct={sim_pct[best_idx]:.3f}")
            self.depth_debug += 1

        self._split_recursive(node.left, X)
        self._split_recursive(node.right, X)

    def _feature_weight(self, f: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Get feature weight(s). Handles scalar int or array of indices."""
        if self._w is None:
            # If no weights, return 1.0 (or array of 1.0s)
            if isinstance(f, np.ndarray):
                return np.ones(f.shape, dtype=float)
            return 1.0
        return self._w[f].astype(float)

    # -------- kinetic-aware helpers --------
    @staticmethod
    def transition_counts(
            labels: np.ndarray,
            tau: int,
            n_states: Optional[int] = None,
            traj_lengths: Optional[Sequence[int]] = None  # <-- ADD THIS
    ) -> np.ndarray:
        """
        Builds a transition count matrix for a given lag time tau.
        If traj_lengths is provided, avoids counting transitions
        across trajectory boundaries.
        """
        labels = labels.astype(int)
        if n_states is None:
            n_states = int(labels.max()) + 1
        C = np.zeros((n_states, n_states), dtype=float)
        T = labels.size

        if traj_lengths is None:
            print("WARNING: traj_lengths not provided to transition_counts(). "
                  "Assuming a single continuous trajectory.")
            for t in range(0, max(0, T - tau)):
                i = labels[t]
                j = labels[t + tau]
                if i >= 0 and j >= 0:
                    C[i, j] += 1
        else:
            if sum(traj_lengths) != T:
                raise ValueError(
                    f"Sum of traj_lengths ({sum(traj_lengths)}) "
                    f"does not match labels.size ({T})"
                )

            start_idx = 0
            for length in traj_lengths:
                end_idx = start_idx + length
                for t in range(start_idx, max(start_idx, end_idx - tau)):
                    i = labels[t]
                    j = labels[t + tau]
                    if i >= 0 and j >= 0:
                        C[i, j] += 1
                start_idx = end_idx

        return C

    @staticmethod
    def transition_matrix(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        row_sums = C.sum(axis=1, keepdims=True) + eps
        return C / row_sums

    def _row_similarity_2x2(self, C: np.ndarray, eps: float = 1e-12) -> float:
        P = self.transition_matrix(C, eps=eps)
        a, b = P[0], P[1]
        num = float(np.dot(a, b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
        return num / den

    def _lag_weight(self, tau: int) -> float:
        if self.lag_mode == "max":
            return 1.0 if tau == max(self.lags) else 0.0
        elif self.lag_mode == "power":
            return float(max(tau, 1) ** self.tau_power)
        else:
            return 1.0

    def _binarize_delta_series(self, S: np.ndarray) -> np.ndarray:
        if S.size == 0:
            return S.astype(np.int8)
        if self.delta_mode == "threshold":
            thr = self.delta_thresh
        else:
            q = np.clip(self.delta_quantile, 0.0, 1.0)
            thr = float(np.quantile(S, q))
        return (S >= thr).astype(np.int8)

    def _split_metrics(
            self, X: np.ndarray, left_mask_full: np.ndarray, pairs_per_lag: Dict[int, np.ndarray],
            split_pair: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Compute node-local metrics for a candidate split, averaged over lags:
          - VAMP-2: predictive correlation of the split variable
          - NMI:    correlation with contact-map change magnitude ΔC
          - SIM:    child row similarity penalty (higher = more redundant)
        """
        has_contacts = self.contact_frames_ is not None
        vamp_wsum = nmi_wsum = sim_wsum = wtot = 0.0

        for tau, P in pairs_per_lag.items():
            per_lag_min = max(1, self.min_pairs // max(len(self.lags), 1))
            if P.shape[0] < per_lag_min:
                continue
            t0, t1 = P[:, 0], P[:, 1]
            x0 = left_mask_full[t0].astype(np.int8)
            x1 = left_mask_full[t1].astype(np.int8)

            w = self._lag_weight(tau)
            if w == 0.0:
                continue

            v2 = _vamp2_binary(x0, x1, ridge=self.ridge)

            if has_contacts:
                if tau in self._deltaC_cache_:
                    S = self._deltaC_cache_[tau][t0]
                else:
                    S = np.array([self._delta_contact_sum(int(a), int(b), split_pair)
                                  for a, b in zip(t0, t1)], dtype=float)
                y = self._binarize_delta_series(S)
                nmi = _nmi_binary(x0, y)
            else:
                nmi = _nmi_binary(x0, x1)

            C = np.zeros((2, 2), dtype=float)
            r0 = (x0 == 1)
            C[0, 0] = np.sum(r0 & (x1 == 1))
            C[0, 1] = np.sum(r0 & (x1 == 0))
            r1 = (x0 == 0)
            C[1, 0] = np.sum(r1 & (x1 == 1))
            C[1, 1] = np.sum(r1 & (x1 == 0))
            sim = self._row_similarity_2x2(C)

            vamp_wsum += w * v2
            nmi_wsum += w * nmi
            sim_wsum += w * sim
            wtot += w

        if wtot == 0.0:
            return 0.0, 0.0, 0.0

        return (vamp_wsum / wtot, nmi_wsum / wtot, sim_wsum / wtot)

    def _subsample_features(self, cand: np.ndarray) -> np.ndarray:
        if self.max_features_to_try is None or cand.size <= self.max_features_to_try:
            return cand
        if not self.stratified or self.feature_types is None:
            return self.rng.choice(cand, size=self.max_features_to_try, replace=False)

        types = np.array(self.feature_types)
        uniq = np.unique(types[cand])
        share = max(1, self.max_features_to_try // max(1, len(uniq)))
        chosen = []
        for t in uniq:
            idx_t = cand[types[cand] == t]
            k = min(share, idx_t.size)
            if k > 0:
                pick = self.rng.choice(idx_t, size=k, replace=False)
                chosen.append(pick)
        chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)
        if chosen.size < self.max_features_to_try:
            pool = np.setdiff1d(cand, chosen)
            extra_k = min(self.max_features_to_try - chosen.size, pool.size)
            if extra_k > 0:
                extra = self.rng.choice(pool, size=extra_k, replace=False)
                chosen = np.concatenate([chosen, extra])
        return np.unique(chosen)

    def _finalize_leaves(self, X: np.ndarray):
        leaf_indices: List[np.ndarray] = []

        def dfs_collect(n: Node):
            if n.is_leaf():
                leaf_indices.append(n.idx)
                return
            dfs_collect(n.left)
            dfs_collect(n.right)

        dfs_collect(self.root_)
        leaf_ids = -np.ones(X.shape[0], dtype=int)
        for i, idx in enumerate(leaf_indices):
            leaf_ids[idx] = i
        self.leaf_indices_ = leaf_indices
        self.leaf_ids_ = leaf_ids

        i_leaf = 0
        def set_ids(n: Node):
            nonlocal i_leaf
            if n.is_leaf():
                n.leaf_id = i_leaf
                i_leaf += 1
                return
            set_ids(n.left)
            set_ids(n.right)
        set_ids(self.root_)

    def rule_to_str(self, r: Rule) -> str:
        name = self.feature_names_[r.feature] if self.feature_names_ else f"feat_{r.feature}"
        return f"{name} is {'present' if r.polarity == 1 else 'absent'}"


# -------------------------
# CLI / Diagnostics
# -------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Path to .pt interaction file")
    parser.add_argument("--out_json", required=True, help="Path to output JSON file")
    parser.add_argument("--lags", type=int, nargs="+", default=[5, 10, 25], help="Time lags")
    parser.add_argument("--alpha", type=float, default=0.7, help="VAMP vs NMI weight")
    parser.add_argument("--min_pairs", type=int, default=200)
    parser.add_argument("--min_leaf", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--max_feats", default="", help='Max features to try, "" means all')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_types_separately", action="store_true")
    parser.add_argument("--present_threshold", type=float, default=0.5)
    parser.add_argument("--py_weights", type=str, default="{}", help="JSON string of type weights")
    parser.add_argument("--apply_hysteresis", action="store_true", help="Debounce binary features with temporal persistence.")
    parser.add_argument("--min_on", type=int, default=3, help="Min consecutive ON frames required to switch 0->1.")
    parser.add_argument("--min_off", type=int, default=3, help="Min consecutive OFF frames required to switch 1->0.")
    # kinetic knobs
    parser.add_argument("--lag_mode", type=str, default="uniform", choices=["uniform", "max", "power"],
                        help="How to weight lags during split scoring: uniform / max-only / power weighting")
    parser.add_argument("--tau_power", type=float, default=1.0, help="Exponent for power weighting over tau (when --lag_mode power)")
    parser.add_argument("--lambda_row", type=float, default=0.25, help="Penalty for kinetically-similar child TPM rows")
    # contact-change inputs
    parser.add_argument("--contact_pkl", type=str, default="", help="Path to soft contact map .pkl (list of frames with (i,j,score))")
    parser.add_argument("--local_window", type=int, default=2, help="Exclude contacts within this residue window of the split pair")
    parser.add_argument("--delta_mode", type=str, default="quantile", choices=["quantile","threshold"], help="How to binarize contact-change magnitude")
    parser.add_argument("--delta_quantile", type=float, default=0.6, help="Quantile for binarizing summed |ΔC| when delta_mode=quantile")
    parser.add_argument("--delta_thresh", type=float, default=0.2, help="Absolute threshold for binarizing summed |ΔC| when delta_mode=threshold")
    parser.add_argument("--contact_weight_json", type=str, default="", help="Optional JSON file specifying active-site residues and weight")

    args = parser.parse_args()

    X, feature_names, feature_types = load_interaction_binary_pt(
        args.pt,
        use_types_separately=args.use_types_separately,
        present_threshold=args.present_threshold,
    )

    if args.apply_hysteresis:
        X = apply_hysteresis(X, min_on=args.min_on, min_off=args.min_off)

    contact_frames = None
    if args.contact_pkl:
        contact_frames = load_contact_map_pkl(args.contact_pkl)

    contact_weight_spec = None
    if args.contact_weight_json:
        with open(args.contact_weight_json, "r") as fh:
            contact_weight_spec = json.load(fh)

    max_feats = None if args.max_feats == "" else int(args.max_feats)
    model = TemporalRuleTreeV6(
        lags=args.lags,
        alpha_vamp=args.alpha,
        min_pairs=args.min_pairs,
        min_leaf=args.min_leaf,
        max_depth=args.max_depth,
        max_features_to_try=max_feats,
        feature_types=feature_types,
        random_state=args.seed,
        lag_mode=args.lag_mode,
        tau_power=args.tau_power,
        lambda_row=args.lambda_row,
        contact_frames=contact_frames,
        local_window=args.local_window,
        delta_mode=args.delta_mode,
        delta_quantile=args.delta_quantile,
        delta_thresh=args.delta_thresh,
        contact_weight_spec=contact_weight_spec,
    )

    type_weight_map = json.loads(args.py_weights)
    if type_weight_map:
        model.set_weights(type_weight_map=type_weight_map)

    model.fit(X, feature_names=feature_names)

    with open(args.out_json, "w") as f:
        json.dump(model.export_tree(), f, indent=2)

    occ = model.occupancy()
    print(f"T={X.shape[0]} frames, F={X.shape[1]} features, Leaves={occ.size}")
    print("Top 5 leaf occupancies:", np.round(np.sort(occ)[::-1][:5], 4))
    print(f"Tree JSON written to: {args.out_json}")

    labels = model.predict_leaf_ids(X)
    out_base = Path(args.out_json)
    out_dir = out_base.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_smooth = 0.5
    def smooth_tpm(C, alpha=0.5):
        K = C.shape[1]
        numer = C + alpha
        denom = C.sum(axis=1, keepdims=True) + alpha * K
        return numer / np.clip(denom, 1e-12, None)

    def eigvals_sorted(P):
        w = np.linalg.eigvals(P.T)
        return np.array(sorted(w, key=lambda z: abs(z), reverse=True))

    def implied_timescales(eigs, tau):
        its = []
        for lam in eigs[1:]:
            if np.isreal(lam):
                lamr = float(np.real(lam))
                if 0 < lamr < 1:
                    its.append(-tau / np.log(lamr))
        return its

    summary = {"lags": []}
    for tau in args.lags:
        C = model.transition_counts(labels, tau)
        P_ml = model.transition_matrix(C)
        P = smooth_tpm(C, alpha=alpha_smooth)

        diag = np.diag(P)
        mean_diag = float(diag.mean())
        min_diag = float(diag.min())
        zero_rows = int(np.sum(C.sum(axis=1) == 0))
        total_pairs = int(C.sum())

        ev = eigvals_sorted(P)
        lam2 = float(np.real(ev[1])) if ev.size > 1 else float("nan")
        gap = float(1.0 - lam2) if np.isfinite(lam2) else float("nan")
        its = implied_timescales(ev, tau)
        its_top3 = [float(x) for x in sorted(its, reverse=True)[:3]]

        print(f"\n=== TPM diagnostics @ lag={tau} ===")
        print(f"pairs={total_pairs} | zero_rows={zero_rows}")
        print(f"mean(diag)={mean_diag:.4f} | min(diag)={min_diag:.4f}")
        print(f"lambda2={lam2:.6f} | spectral_gap={gap:.6f}")
        if its_top3:
            print(f"top implied timescales (frames): {', '.join(f'{x:.1f}' for x in its_top3)}")
        else:
            print("top implied timescales: (none finite)")

        try:
            import pandas as _pd
            _pd.DataFrame(C).to_csv(out_dir / f"{out_base.stem}_counts_tau{tau}.csv", index=False)
            _pd.DataFrame(P).to_csv(out_dir / f"{out_base.stem}_tpm_smoothed_tau{tau}.csv", index=False)
            _pd.DataFrame(P_ml).to_csv(out_dir / f"{out_base.stem}_tpm_mle_tau{tau}.csv", index=False)
        except Exception:
            pass

        summary["lags"].append({
            "tau": int(tau),
            "pairs": total_pairs,
            "zero_rows": zero_rows,
            "mean_diag": mean_diag,
            "min_diag": min_diag,
            "lambda2": lam2,
            "spectral_gap": gap,
            "its_top3_frames": its_top3,
        })

    with open(out_dir / f"{out_base.stem}_tpm_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
