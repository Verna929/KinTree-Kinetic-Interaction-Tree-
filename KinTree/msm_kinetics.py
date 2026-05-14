import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig, eigvals
from collections import defaultdict, Counter
from temporal_rule_tree import TemporalRuleTreeV6
from igme import IGME
from scipy.linalg import expm
try:
    from sklearn.cluster import KMeans
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ---------- hysteresis helpers ----------
def _run_lengths_binary(seq):
    """
    Given a 1D binary array (0/1), return lists of ON-lengths and OFF-lengths.
    """
    seq = (seq > 0).astype(np.int8)
    if seq.size == 0:
        return [], []

    on_lengths = []
    off_lengths = []

    curr_val = seq[0]
    length = 1
    for x in seq[1:]:
        if x == curr_val:
            length += 1
        else:
            if curr_val == 1:
                on_lengths.append(length)
            else:
                off_lengths.append(length)
            curr_val = x
            length = 1
    if curr_val == 1:
        on_lengths.append(length)
    else:
        off_lengths.append(length)

    return on_lengths, off_lengths

def _iter_segments(T, traj_lengths=None):
    """
    Yield (start, end) indices for each trajectory segment.
    """
    if traj_lengths is None:
        yield 0, T
        return

    total = sum(traj_lengths)
    if total != T:
        raise ValueError(f"Sum(traj_lengths)={total}, but T={T}")
    start = 0
    for L in traj_lengths:
        yield start, start + L
        start += L

def suggest_global_hysteresis_thresholds(
        X,
        traj_lengths=None,
        on_percentile=75,
        off_percentile=75,
        min_clip=1,
        max_clip=None,
        require_activity=True
):
    """
    Compute global min_on/min_off, excluding features that never turn ON
    or never turn OFF (optional).
    """
    X = (X > 0).astype(np.int8)
    T, F = X.shape
    all_on, all_off = [], []

    for s, e in _iter_segments(T, traj_lengths):
        seg = X[s:e, :]
        for f in range(F):
            seq = seg[:, f]
            on_lengths, off_lengths = _run_lengths_binary(seq)
            if require_activity:
                if np.sum(seq) == 0 or np.sum(seq) == seq.size:
                    continue
            all_on.extend(on_lengths)
            all_off.extend(off_lengths)

    all_on = np.array(all_on, dtype=float)
    all_off = np.array(all_off, dtype=float)

    if len(all_on) == 0 or len(all_off) == 0:
        raise ValueError("No valid ON/OFF runs found after filtering.")

    min_on = np.percentile(all_on, on_percentile)
    min_off = np.percentile(all_off, off_percentile)

    if max_clip is None:
        max_clip = T

    min_on = int(np.clip(min_on, min_clip, max_clip))
    min_off = int(np.clip(min_off, min_clip, max_clip))

    print(f"[auto] Suggested thresholds: min_on={min_on}, min_off={min_off}")
    print(f"  (computed from {len(all_on)} ON runs, {len(all_off)} OFF runs)")
    return min_on, min_off, all_on, all_off

# ---------- basic helpers for pre-scan ----------

def _build_P(labels, tau, alpha=1.0, traj_lengths=None):  # <-- same signature
    """
    Build a smoothed TPM P(τ) and return (P, total_pairs).
    Uses the unified _smooth_tpm() for Dirichlet-α smoothing.
    """
    labels = np.asarray(labels, dtype=int)
    n = int(labels.max()) + 1
    C = np.zeros((n, n), dtype=float)
    T = len(labels)

    if traj_lengths is None:
        for t in range(T - tau):
            i = labels[t]
            j = labels[t + tau]
            if i >= 0 and j >= 0:
                C[i, j] += 1.0
    else:
        start_idx = 0
        for length in traj_lengths:
            end_idx = start_idx + length
            for t in range(start_idx, max(start_idx, end_idx - tau)):
                i = labels[t]
                j = labels[t + tau]
                if i >= 0 and j >= 0:
                    C[i, j] += 1.0
            start_idx = end_idx

    P = _smooth_tpm(C, alpha=alpha)
    return P, int(C.sum())

def _its_from_P(P, tau, top=3):
    w = np.linalg.eigvals(P.T).real
    w = np.sort(w)[::-1]
    its = [-tau / np.log(l) for l in w[1:] if 0.0 < l < 1.0]
    return its[:top]

def _ck_error(P_tau, P_2tau):
    P2 = P_tau @ P_tau
    num = np.linalg.norm(P_2tau - P2, ord='fro')
    den = max(1e-12, np.linalg.norm(P_2tau, ord='fro'))
    return num / den

def prescan_tau(labels, taus, alpha=1.0, top_modes=3, traj_lengths=None): # <-- ADDED ARG
    out = {}
    for tau in taus:
        P_tau, pairs_tau = _build_P(labels, tau, alpha=alpha, traj_lengths=traj_lengths)
        P_2tau, pairs_2tau = _build_P(labels, 2 * tau, alpha=alpha, traj_lengths=traj_lengths)
        its = _its_from_P(P_tau, tau, top=top_modes)
        ck = _ck_error(P_tau, P_2tau)
        out[tau] = {"ITS": its, "CK": ck, "pairs_tau": pairs_tau, "pairs_2tau": pairs_2tau}
    return out

def suggest_tau_range(diag, ck_tol=0.10):
    ok = [t for t, d in diag.items() if d["CK"] < ck_tol]
    return (min(ok), max(ok)) if ok else None

def suggest_tau_star(diag, its_tol=0.10, ck_tol=0.10, top=2):
    taus = list(diag.keys())
    taus.sort()
    for a, b in zip(taus[:-1], taus[1:]):
        Ia, Ib = diag[a]["ITS"], diag[b]["ITS"]
        if not Ia or not Ib:
            continue
        m = min(top, len(Ia), len(Ib))
        its_ok = all(abs(Ib[k] - Ia[k]) / max(Ia[k], 1e-12) < its_tol for k in range(m))
        ck_ok = diag[b]["CK"] < ck_tol
        if its_ok and ck_ok:
            return b
    return None

def _smooth_tpm(C, alpha=0.5, eps=1e-12):
    """
    Unified symmetric Dirichlet-α smoothing.
    Converts transition counts C into a TPM P.

        P[i,j] = (C[i,j] + alpha) / (sum_j C[i,j] + alpha * K)

    Parameters
    ----------
    C : array-like (n_states, n_states)
        Transition counts.
    alpha : float
        Dirichlet prior strength. Use alpha=0.5 for weak smoothing;
        alpha=1.0 for stronger smoothing.
    eps : float
        Numerical stability constant.

    Returns
    -------
    P : ndarray
        Smoothed transition probability matrix.
    """
    C = np.asarray(C, dtype=float)
    K = C.shape[1]

    numer = C + alpha
    denom = C.sum(axis=1, keepdims=True) + alpha * K
    denom = np.clip(denom, eps, None)

    return numer / denom

# ---------- IGME helpers ----------

def collect_counts_multi_tau(labels, taus, n_states=None, traj_lengths=None):
    """
    Use TemporalRuleTreeV6.transition_counts(...) to build a dict
    {tau -> C(tau)} with boundary-aware counting.
    """
    labels = np.asarray(labels, int)
    if n_states is None:
        n_states = int(labels.max()) + 1

    C_dict = {}
    for tau in sorted(taus):
        C = TemporalRuleTreeV6.transition_counts(
            labels, tau, n_states=n_states, traj_lengths=traj_lengths
        )
        C_dict[tau] = C
    return C_dict

def build_tpm_series_from_counts(C_tau, alpha_smooth=0.5):
    """
    C_tau: dict {tau -> C(tau)}.
    Returns:
      taus_sorted: [τ1,...,τM]
      T_series:    array shape (M, n, n) with TPM(τk)
    """
    taus_sorted = sorted(C_tau.keys())
    T_list = [_smooth_tpm(C_tau[tau], alpha=alpha_smooth) for tau in taus_sorted]
    T_series = np.stack(T_list, axis=0)
    return np.array(taus_sorted), T_series

def igme_fit_from_tpm_series(T_series, begin=2, end=None, stride=1,
                             min_its=0.0, max_its=1e18, log_approx_order=0):
    """
    Run IGME.scan on a series of TPMs and return the 'best' IGME model.
    begin/end are indices on the *time-index* of T_series (1-based, as IGME wants).
    """
    n_lags = T_series.shape[0]
    if end is None:
        end = n_lags

    igme = IGME(logarithm_approx_order=log_approx_order)
    scan_out = igme.scan(T_series, begin=begin, end=end, stride=stride,
                         rmse_weighted_by_sp=True, debug=False)

    best = igme.top_model(scan_out, n=1, min_its=min_its, max_its=max_its)
    print(f"[IGME] best rmse = {best.rmse}, window = [{best.begin}, {best.end}]")
    return best

def spectral_lumping_from_Th(Th, n_macrostates, n_init=50, random_state=0):
    """
    Kinetic lumping using the IGME long-time propagator Th.
    - Compute leading right eigenvectors of Th^T
    - Cluster rows of the eigenvector matrix with KMeans
    Returns:
      micro_to_macro: array of shape (n_micro,) mapping each microstate → macrostate id
    """
    Th = np.asarray(Th, float)
    w, V = np.linalg.eig(Th.T)
    idx = np.argsort(-np.abs(w))
    w = w[idx]
    V = V[:, idx]

    k = max(1, n_macrostates - 1)
    eigvecs = np.real(V[:, 1:1+k])

    norms = np.linalg.norm(eigvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_emb = eigvecs / norms

    kmeans = KMeans(n_clusters=n_macrostates, n_init=n_init, random_state=random_state)
    micro_to_macro = kmeans.fit_predict(X_emb)
    return micro_to_macro

def ck_test_macro(
    macro_labels, tau0, k_list,
    traj_lengths=None, alpha_smooth=0.5,
    state_names=None
):
    """
    CK test for macro MSM with 95% confidence intervals on P_data(ii).
    """

    macro_labels = np.asarray(macro_labels, int)
    n_mac = int(macro_labels.max()) + 1
    C_base = TemporalRuleTreeV6.transition_counts(
        macro_labels, tau0, n_states=n_mac, traj_lengths=traj_lengths
    )
    P_base = _smooth_tpm(C_base, alpha=alpha_smooth)
    P_model_k = {k: np.linalg.matrix_power(P_base, k) for k in k_list}


    P_data_k = {}
    C_data_k = {}

    for k in k_list:
        tau_k = k * tau0
        C_k = TemporalRuleTreeV6.transition_counts(
            macro_labels, tau_k, n_states=n_mac, traj_lengths=traj_lengths
        )
        P_k = _smooth_tpm(C_k, alpha=alpha_smooth)
        P_data_k[k] = P_k
        C_data_k[k] = C_k

    fig, axes = plt.subplots(n_mac, 1, figsize=(5, 3*n_mac), sharex=True)
    if n_mac == 1:
        axes = [axes]

    lags_phys = np.array(k_list) * tau0

    for i in range(n_mac):
        ax = axes[i]
        data_vals = np.array([P_data_k[k][i, i] for k in k_list])
        model_vals = np.array([P_model_k[k][i, i] for k in k_list])
        N_vals = np.array([C_data_k[k][i].sum() for k in k_list])
        p = data_vals
        N = np.maximum(N_vals, 1)

        sigma = np.sqrt(p * (1 - p) / N)
        lower = np.clip(p - 1.96 * sigma, 0, 1)
        upper = np.clip(p + 1.96 * sigma, 0, 1)

        ax.fill_between(
            lags_phys,
            lower,
            upper,
            color="C0",
            alpha=0.25,
            label="95% CI (data)"
        )
        ax.plot(lags_phys, p, "o-", color="C0", label="data")
        ax.plot(lags_phys, model_vals, "s--", color="C1", label="MSM")
        name = state_names[i] if state_names is not None else f"State {i+1}"
        ax.set_ylabel(f"{name}\nP(ii)")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend()

    axes[-1].set_xlabel(f"lag (frames, base={tau0})")
    fig.tight_layout()
    return fig, axes


def _build_macro_projection_matrix(macro_labels, n_macro=None):
    macro_labels = np.asarray(macro_labels, dtype=int)
    n_micro = macro_labels.size
    if n_macro is None:
        n_macro = int(macro_labels.max()) + 1
    S = np.zeros((n_micro, n_macro))
    for i, k in enumerate(macro_labels):
        S[i, k] = 1.0
    return S

def clean_labels(labels):
    """Convert list of strings like 'S0' → integer 0."""
    arr = np.array(labels)
    if arr.dtype.kind in {'U', 'S', 'O'}:
        cleaned = np.array([int(''.join(filter(str.isdigit, str(x)))) for x in arr])
        return cleaned
    return arr.astype(int)

def derive_micro_to_macro(labels, macro_labels):
    """
    Determine which macrostate each microstate belongs to
    by majority vote over all frames.
    """
    labels = np.asarray(labels, dtype=int)
    macro_labels = np.asarray(macro_labels, dtype=int)
    n_micro = labels.max() + 1
    micro_to_macro = np.zeros(n_micro, dtype=int)

    for i in range(n_micro):
        mask = (labels == i)
        if np.any(mask):
            counts = np.bincount(macro_labels[mask])
            micro_to_macro[i] = np.argmax(counts)
        else:
            micro_to_macro[i] = -1  # unused microstate
    return micro_to_macro

def coarse_grain_transition_matrix(Th, pi_micro, micro_to_macro):
    """
    Lump microstate transition matrix (Th) into macrostate matrix.
    """
    micro_to_macro = np.asarray(micro_to_macro, dtype=int)
    n_macro = micro_to_macro[micro_to_macro >= 0].max() + 1
    n_micro = Th.shape[0]

    # build S
    S = np.zeros((n_micro, n_macro))
    for i, k in enumerate(micro_to_macro):
        if k >= 0:
            S[i, k] = 1.0

    Pi = np.diag(pi_micro)
    A = S.T @ Pi @ S
    B = S.T @ Pi @ Th @ S
    T_macro = np.linalg.solve(A, B)
    T_macro /= T_macro.sum(axis=1, keepdims=True)
    return T_macro

def mfpt_matrix_igme(Th):
    """
    Compute MFPTs (mean first passage times) between all states
    using the IGME effective transition matrix Th.
    Equivalent to MSM MFPT but uses IGME-corrected dynamics.
    """
    Th = np.array(Th, dtype=float)
    n = Th.shape[0]
    w, V = eig(Th.T)
    k = int(np.argmax(np.real(w)))
    pi = np.real(V[:, k])
    pi = np.maximum(pi, 0)
    if pi.sum() == 0:
        pi = np.ones(n)
    pi /= pi.sum()

    I = np.eye(n)
    Z = np.linalg.inv(I - Th + np.outer(np.ones(n), pi))

    T = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                T[i, j] = 0.0
            else:
                T[i, j] = (Z[j, j] - Z[i, j]) / max(pi[j], 1e-12)
    return T


def print_mfpt_table_igme(Th, state_labels=None, precision=1):
    """
    Print MFPT matrix from IGME transition matrix in a readable table.
    """
    T = mfpt_matrix_igme(Th)
    n = T.shape[0]
    if state_labels is None:
        state_labels = [str(i) for i in range(n)]
    header = "     " + " ".join(f"{lbl:>8}" for lbl in state_labels)
    print(header)
    for i, row in enumerate(T):
        line = f"{state_labels[i]:>3} " + " ".join(f"{val:8.{precision}f}" for val in row)
        print(line)
    return T


def plot_mfpt_matrix_igme(T, tau=None):
    """
    Plot IGME MFPT matrix as a heatmap.
    """
    plt.figure(figsize=(5, 4))
    im = plt.imshow(T, cmap="viridis", origin="lower")
    plt.colorbar(im, label="MFPT (frames)")
    title = f"IGME MFPTs (τ={tau})" if tau else "IGME MFPTs"
    plt.title(title)
    plt.xlabel("Target state")
    plt.ylabel("Start state")
    plt.tight_layout()
    plt.show()

def get_stationary_from_igme(igme):
    """
    Return stationary distribution from an IGME model safely.
    """
    if hasattr(igme, "p_eq"):
        pi = np.array(igme.p_eq).flatten()
    elif hasattr(igme, "Peq"):
        pi = np.array(igme.Peq).flatten()
    elif hasattr(igme, "Th"):
        w, V = np.linalg.eig(igme.Th.T)
        k = int(np.argmax(np.real(w)))
        pi = np.real(V[:, k])
        pi = np.maximum(pi, 0)
        pi /= pi.sum()
    else:
        raise AttributeError("IGME object has no p_eq / Peq / Th attribute to get stationary distribution.")
    return pi

# ---------- MSM helpers ----------
def collect_split_features(node, model, feats=None):
    if feats is None: feats = []
    if node.get("leaf", False): return feats
    rule = node["rule"]
    feats.append(rule["name"])
    collect_split_features(node["left"], model, feats)
    collect_split_features(node["right"], model, feats)
    return feats

def summarize_rules_per_macro_weighted(leaf_paths, leaf_to_macro, leaf_indices, top_k=10):
    """
    Aggregate rule usage across leaves belonging to each macrostate,
    weighted by leaf occupancy (#frames in that leaf).

    Parameters
    ----------
    leaf_paths : dict
        leaf_id -> list of (feature_name, polarity, took_left_branch)
    leaf_to_macro : dict
        leaf_id -> macro_id
    leaf_indices : list of np.ndarray
        from model.leaf_indices_ (frames per leaf)
    top_k : int
        how many top rules to report per macro

    Returns
    -------
    macro_top : dict
        macro_id -> list of (rule_str, weighted_count)
    """
    from collections import defaultdict, Counter
    macro_rules = defaultdict(Counter)

    for leaf_id, path in leaf_paths.items():
        m = int(leaf_to_macro.get(leaf_id, -1))
        if m < 0:
            continue
        weight = len(leaf_indices[leaf_id])
        for feat_name, pol, took_left in path:
            cond_left = (pol == 1)
            cond = cond_left if took_left else (not cond_left)
            cond_str = f"{feat_name}={'present' if cond else 'absent'}"
            macro_rules[m][cond_str] += weight

    macro_top = {m: counter.most_common(top_k) for m, counter in macro_rules.items()}
    return macro_top

def _implied_timescales_from_P(P, tau):
    w = np.real(eigvals(P.T))
    its = []
    for lam in np.sort(w)[::-1][1:]:
        if 0 < lam < 1:
            its.append(-tau / np.log(lam))
    return its

def mfpt_submatrix(P, i, j):
    n = P.shape[0]
    S = [k for k in range(n) if k != j]
    P_S = P[np.ix_(S, S)]
    A = np.eye(len(S)) - P_S
    e = np.ones(len(S))
    t_S = np.linalg.lstsq(A, e, rcond=None)[0]
    return float(t_S[S.index(i)])

def largest_scc_indices(P, eps=1e-12):
    A = (P > eps).astype(int)
    n = A.shape[0]
    R = A.copy(); np.fill_diagonal(R, 1)
    for k in range(n):
        R = np.logical_or(R, (R[:, [k]] & R[[k], :]))
    comps = []
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        if used[i]: continue
        comp = np.where(np.logical_and(R[i], R[:, i]))[0]
        used[comp] = True
        comps.append(comp)
    if not comps:
        return np.array([], dtype=int)
    comp = max(comps, key=lambda x: x.size)
    return np.array(sorted(list(comp)), dtype=int)

def reversible_projection(P):
    w, V = eig(P.T)
    k = int(np.argmax(np.real(w)))
    pi = np.real(V[:, k]); pi = np.maximum(pi, 0)
    if pi.sum() == 0: pi = np.ones(P.shape[0])
    pi = pi / pi.sum()
    S = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            num = pi[i]*P[i,j] + pi[j]*P[j,i]
            S[i,j] = num / (2.0 * max(pi[i], 1e-12))
    S = S / np.clip(S.sum(axis=1, keepdims=True), 1e-12, None)
    return S

def micro_counts_from_labels(labels, tau, n_states=None):
    labels = np.asarray(labels, dtype=int)
    if n_states is None:
        n_states = int(labels.max()) + 1
    C = np.zeros((n_states, n_states), dtype=float)
    T = labels.size
    for t in range(T - tau):
        i = labels[t]; j = labels[t + tau]
        if i >= 0 and j >= 0:
            C[i, j] += 1.0
    return C

def collapse_labels(labels_micro, z_micro2macro):
    L = int(labels_micro.max()) + 1
    z = np.asarray(z_micro2macro, dtype=int)
    out = np.where(labels_micro >= 0, z[labels_micro], -1)
    return out

# ---------- Spectral lumping ----------
def spectral_lumping_from_counts(C_micro, k_mac=8, alpha=1.0, random_state=0):
    L = C_micro.shape[0]
    if L <= k_mac:
        z = np.arange(L) % k_mac
    P = _smooth_tpm(C_micro, alpha=alpha)

    w, V = eig(P.T)
    k = int(np.argmax(np.real(w)))
    pi = np.real(V[:, k]); pi = np.maximum(pi, 1e-12); pi = pi/pi.sum()
    S = np.diag(np.sqrt(pi)) @ P @ np.diag(1/np.sqrt(pi))
    S = 0.5*(S+S.T)

    wS, VS = eig(S)
    idx = np.argsort(-np.real(wS))
    U = np.real(VS[:, idx[1:1+k_mac]])

    if _HAS_SK:
        z = KMeans(n_clusters=k_mac, n_init=20, random_state=random_state).fit_predict(U)
    else:
        rng = np.random.default_rng(random_state)
        cent = U[rng.choice(U.shape[0], size=k_mac, replace=False)]
        for _ in range(50):
            d2 = ((U[:, None, :] - cent[None, :, :])**2).sum(axis=2)
            z = d2.argmin(axis=1)
            for c in range(k_mac):
                pts = U[z==c]
                if pts.size: cent[c]=pts.mean(axis=0)

    C_macro = np.zeros((k_mac, k_mac), float)
    for i in range(L):
        for j in range(L):
            C_macro[z[i], z[j]] += C_micro[i,j]
    P_macro = _smooth_tpm(C_macro, alpha=alpha)
    return z, P, P_macro, C_macro

# ---------- Rule attribution ----------
def collect_leaf_paths(tree_dict):
    paths = {}
    def dfs(node, path):
        if node.get("leaf", False):
            paths[int(node["leaf_id"])] = list(path)
            return
        rule = node["rule"]
        feat_name = rule["name"]
        pol = int(rule.get("polarity_left_is_present", 1))
        dfs(node["left"],  path + [(feat_name, pol, True)])
        dfs(node["right"], path + [(feat_name, pol, False)])
    dfs(tree_dict, [])
    return paths

def summarize_rules_per_macro_percent(leaf_paths, leaf_to_macro, leaf_indices, top_k=10):
    macro_rules = defaultdict(Counter)
    macro_totals = Counter()
    for leaf_id, path in leaf_paths.items():
        m = int(leaf_to_macro.get(leaf_id, -1))
        if m < 0: continue
        weight = len(leaf_indices[leaf_id])
        macro_totals[m] += weight
        for feat_name, pol, took_left in path:
            cond_left = (pol==1)
            cond = cond_left if took_left else (not cond_left)
            cond_str = f"{feat_name}={'present' if cond else 'absent'}"
            macro_rules[m][cond_str] += weight
    macro_top = {}
    for m, counter in macro_rules.items():
        total = macro_totals[m]
        items = [(rule, 100.0*count/total) for rule,count in counter.most_common(top_k)]
        macro_top[m]=items
    return macro_top

def plot_macro_its(results):
    taus = sorted(results.keys())
    max_its = max(len(res["ITS"]) for res in results.values())
    its_mat = np.full((len(taus), max_its), np.nan)
    for i, tau in enumerate(taus):
        its = results[tau]["ITS"]
        for k in range(len(its)):
            its_mat[i, k] = its[k]
    plt.figure(figsize=(6,4))
    for k in range(max_its):
        plt.plot(taus, its_mat[:,k], marker="o", label=f"ITS{k+1}")
    plt.xlabel("Lag time τ")
    plt.ylabel("Implied timescale (frames)")
    plt.title("Macrostate ITS vs τ")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_macro_ck(labels_macro, model, tau, alpha=1.0, states=None, max_mult=5, traj_lengths=None): # <-- ADDED
    n_states = int(labels_macro.max())+1
    C_tau = model.transition_counts(labels_macro, tau, n_states=n_states, traj_lengths=traj_lengths) # <-- FIXED
    P_tau = _smooth_tpm(C_tau, alpha)
    if states is None:
        states = list(range(min(3,n_states)))
    plt.figure(figsize=(6,4))
    for s in states:
        obs, pred = [], []
        for k in range(1,max_mult+1):
            C_k = model.transition_counts(labels_macro, k*tau, n_states=n_states, traj_lengths=traj_lengths)
            P_k = _smooth_tpm(C_k, alpha)
            obs.append(P_k[s,s])
            P_pred = np.linalg.matrix_power(P_tau, k)
            pred.append(P_pred[s,s])
        plt.plot(range(1,max_mult+1), obs, "o-", label=f"State {s} obs")
        plt.plot(range(1,max_mult+1), pred, "--", label=f"State {s} pred")
    plt.xlabel("Multiple of τ")
    plt.ylabel("Self-transition probability")
    plt.title(f"CK test (τ={tau})")
    plt.legend(); plt.tight_layout(); plt.show()

def macro_pipeline(
    model, labels_micro, tree_dict,
    traj_lengths=None,
    tau_ref = None, # this should be the sane with the model lag time
    k_mac=8, taus_multi=[1,2,3,4,5],
    alpha_counts=1.0, alpha_validate=1.0,
    enforce_reversible=True, random_state=0
):
    taus=tau_ref * np.asarray(taus_multi)
    L=int(labels_micro.max())+1
    C_micro = model.transition_counts(labels_micro, tau_ref, n_states=L, traj_lengths=traj_lengths)
    z_micro2macro, P_micro, P_macro_ref, C_macro_ref = spectral_lumping_from_counts(
        C_micro,k_mac=k_mac,alpha=alpha_counts,random_state=random_state
    )
    labels_macro=collapse_labels(labels_micro,z_micro2macro)
    print(f"[Lumping] microstates={L} -> macrostates={k_mac}")

    results={}
    for tau in taus:
        C_mac = model.transition_counts(labels_macro, tau, n_states=k_mac, traj_lengths=traj_lengths)
        P_mac=_smooth_tpm(C_mac,alpha=alpha_validate)
        keep=largest_scc_indices(P_mac)
        P_red=P_mac[np.ix_(keep,keep)]
        if enforce_reversible: P_red=reversible_projection(P_red)
        its=_implied_timescales_from_P(P_red,tau)
        results[tau]={"P_macro":P_mac,"ITS":its,"pairs":int(C_mac.sum())}
        print(f"[Macro validation] τ={tau} | states={P_red.shape[0]} | pairs={int(C_mac.sum())}")
        print("Top ITS:",[round(x,2) for x in its[:5]])

    leaf_paths=collect_leaf_paths(tree_dict)
    leaf_to_macro={leaf_id:int(z_micro2macro[leaf_id]) for leaf_id in range(len(z_micro2macro))}
    macro_rules_top=summarize_rules_per_macro_percent(leaf_paths,leaf_to_macro,model.leaf_indices_,top_k=12)

    plot_macro_its(results)
    tau_ck=tau_ref
    states_num = list(range(k_mac))
    plot_macro_ck(labels_macro, model, tau=tau_ck, alpha=alpha_validate, states=states_num, traj_lengths=traj_lengths)

    return {"z_micro2macro":z_micro2macro,"labels_macro":labels_macro,
            "macro_validation":results,"macro_rules_top":macro_rules_top}

def plot_igme_ck_test(igme_result, T_series, taus, state_indices=None):
    """
    Plot the IGME-corrected CK test.

    Parameters
    ----------
    igme_result : IGME object
        The result from igme.top_model() or igme.scan().
        Must have .lnA and .lnTh attributes.
    T_series : np.ndarray
        The actual data TPMs at each lag time (from build_tpm_series_from_counts).
        Shape: (n_lags, n_states, n_states)
    taus : np.ndarray
        The physical lag times corresponding to T_series.
    state_indices : list of int
        Which states to plot diagonal decay for (e.g. [0, 1, 2]).
    """
    import matplotlib.pyplot as plt

    lnA = igme_result.lnA
    lnTh = igme_result.lnTh
    n_states = lnTh.shape[0]

    if state_indices is None:
        state_indices = range(min(4, n_states))

    fig, axes = plt.subplots(len(state_indices), 1, figsize=(6, 3 * len(state_indices)), sharex=True)
    if len(state_indices) == 1: axes = [axes]

    t_smooth = np.linspace(taus.min(), taus.max(), 100)
    pred_vals = {i: [] for i in state_indices}

    for t in t_smooth:
        LogT = lnA + t * lnTh
        T_pred = expm(LogT)
        for i in state_indices:
            pred_vals[i].append(T_pred[i, i])

    data_vals = {i: [] for i in state_indices}
    for k, t in enumerate(taus):
        P_data = T_series[k]
        for i in state_indices:
            data_vals[i].append(P_data[i, i])

    for idx, state_idx in enumerate(state_indices):
        ax = axes[idx]
        ax.plot(taus, data_vals[state_idx], 'o', color='black', label='MD Data', alpha=0.7)
        ax.plot(t_smooth, pred_vals[state_idx], '--', color='red', linewidth=2, label='IGME Model')
        ax.set_ylabel(f"Probability P({state_idx} -> {state_idx})")
        ax.set_title(f"State {state_idx} Relaxation")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    axes[-1].set_xlabel("Lag Time (frames)")
    plt.tight_layout()
    plt.show()

def macro_feature_presence(X, feature_names, labels_macro, split_features):
    """
    Compute percentage of frames where each split feature is present in each macrostate.
    Returns a DataFrame [macro × feature].
    """
    labels_macro = np.asarray(labels_macro, dtype=int)
    n_mac = int(labels_macro.max()) + 1
    idxs = [feature_names.index(f) for f in split_features if f in feature_names]
    feats = [feature_names[i] for i in idxs]

    data = np.zeros((n_mac, len(feats)))
    for m in range(n_mac):
        mask = labels_macro == m
        if not np.any(mask):
            continue
        Xm = X[mask][:, idxs]
        data[m] = Xm.mean(axis=0) * 100.0  # convert to percentage

    df = pd.DataFrame(data, columns=feats)
    df.index = [f"Macro {m}" for m in range(n_mac)]
    return df.round(2)

def mfpt_matrix(P):
    """
    Compute mean first passage times (MFPT) between all states of a Markov chain.
    P : (n,n) row-stochastic transition matrix.
    Returns: T (n,n) array with MFPT[i,j].
    """
    n = P.shape[0]
    # stationary distribution
    w, V = eig(P.T)
    k = int(np.argmax(np.real(w)))
    pi = np.real(V[:, k])
    pi = np.maximum(pi, 0)
    if pi.sum() == 0:
        pi = np.ones(n)
    pi /= pi.sum()

    I = np.eye(n)
    Z = np.linalg.inv(I - P + np.outer(np.ones(n), pi))
    T = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                T[i,j] = 0.0
            else:
                T[i,j] = (Z[j,j] - Z[i,j]) / max(pi[j], 1e-12)
    return T

def print_mfpt_table(P, state_labels=None, precision=1):
    """
    Print MFPT matrix as a nicely formatted table.
    """
    T = mfpt_matrix(P)
    n = T.shape[0]
    if state_labels is None:
        state_labels = [str(i) for i in range(n)]
    header = "     " + " ".join(f"{lbl:>8}" for lbl in state_labels)
    print(header)
    for i, row in enumerate(T):
        line = f"{state_labels[i]:>3} " + " ".join(f"{val:8.{precision}f}" for val in row)
        print(line)
    return T

def plot_mfpt_matrix(T, tau=None, outputpath='MFPT.png', dpi=600):
    """
    Visualize MFPT matrix as a heatmap.
    """
    plt.figure(figsize=(5,4))
    im = plt.imshow(T, cmap="viridis", origin="lower")
    plt.colorbar(im, label="MFPT (frames)")
    if tau:
        plt.title(f"Macro MFPTs (τ={tau})")
    else:
        plt.title("Macro MFPTs")
    plt.xlabel("Target state")
    plt.ylabel("Start state")
    plt.tight_layout()
    plt.savefig(outputpath, dpi=dpi, bbox_inches="tight")
    plt.show()

def calculate_tpt_flux(T, pi, source, sink):
    n = len(T)
    A_mat = np.eye(n) - T
    b = np.zeros(n)

    A_mat[source, :] = 0; A_mat[source, source] = 1; b[source] = 0
    A_mat[sink, :] = 0; A_mat[sink, sink] = 1; b[sink] = 1

    q_plus = np.linalg.solve(A_mat, b)
    q_minus = 1 - q_plus

    flux = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                f = pi[i] * q_minus[i] * T[i, j] * q_plus[j]
                f_rev = pi[j] * q_minus[j] * T[j, i] * q_plus[i]
                flux[i, j] = max(0, f - f_rev)

    return flux, q_plus

def get_path_summary(flux_matrix, source, sink, labels):
    total_flux = flux_matrix[source, :].sum()
    remaining_flux = flux_matrix.copy()
    paths = []

    for _ in range(10):  # Top 10 paths
        if remaining_flux[source, :].sum() < 1e-15:
            break

        path = [source]
        current = source
        capacity = np.inf

        while current != sink:
            next_node = np.argmax(remaining_flux[current, :])
            step_flux = remaining_flux[current, next_node]
            if step_flux <= 0: break

            capacity = min(capacity, step_flux)
            path.append(next_node)
            current = next_node
            if current in path[:-1]: break

        if path[-1] != sink: break

        percentage = capacity / total_flux
        paths.append((percentage, [labels[i] for i in path]))
        for i in range(len(path) - 1):
            remaining_flux[path[i], path[i+1]] -= capacity

    return paths