"""统计与绘图工具函数。"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def fdr_correct(pvalues: Iterable[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(list(pvalues), dtype=float)
    mask = ~np.isnan(p)
    reject = np.full(len(p), False)
    qvals = np.full(len(p), np.nan)
    if mask.sum() == 0:
        return reject, qvals
    r, q, _, _ = multipletests(p[mask], alpha=alpha, method="fdr_bh")
    reject[mask] = r
    qvals[mask] = q
    return reject, qvals


def compare_continuous(x: pd.Series, y: pd.Series) -> tuple[float, str]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 3 or len(y) < 3:
        return np.nan, "insufficient"
    _, p_norm_x = stats.shapiro(x) if len(x) <= 5000 else (0, 1)
    _, p_norm_y = stats.shapiro(y) if len(y) <= 5000 else (0, 1)
    if p_norm_x > 0.05 and p_norm_y > 0.05:
        _, p = stats.ttest_ind(x, y, equal_var=False)
        return p, "t-test"
    _, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return p, "mann-whitney"


def compare_categorical(table: pd.DataFrame) -> tuple[float, str]:
    if table.shape == (2, 2) and table.values.min() >= 0:
        if table.values.min() < 5:
            _, p = stats.fisher_exact(table.values)
            return p, "fisher"
    _, p, _, _ = stats.chi2_contingency(table)
    return p, "chi2"


def spearman_with_fdr(df: pd.DataFrame, rows: list[str], cols: list[str], alpha: float = 0.05):
    records = []
    pvals = []
    for r in rows:
        for c in cols:
            sub = df[[r, c]].dropna()
            if len(sub) < 5:
                rho, p = np.nan, np.nan
            else:
                rho, p = stats.spearmanr(sub[r], sub[c])
            records.append({"row": r, "col": c, "rho": rho, "p": p})
            pvals.append(p)
    out = pd.DataFrame(records)
    _, q = fdr_correct(pvals, alpha=alpha)
    out["q"] = q
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(np.mean(sample))
    low = np.quantile(boots, (1 - ci) / 2)
    high = np.quantile(boots, 1 - (1 - ci) / 2)
    return float(np.mean(values)), float(low), float(high)


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 1000, ci: float = 0.95, seed: int = 42):
    """对 (y_true, y_score) 对做 bootstrap，估计 AUC 置信区间。"""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    idx = np.arange(len(y_true))
    boots = []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2:
            continue
        boots.append(roc_auc_score(y_true[b], y_score[b]))
    if not boots:
        point = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
        return point, np.nan, np.nan
    point = roc_auc_score(y_true, y_score)
    low = np.quantile(boots, (1 - ci) / 2)
    high = np.quantile(boots, 1 - (1 - ci) / 2)
    return float(point), float(low), float(high)


def permutation_auc_pvalue(y_true, y_score, n_perm: int = 500, seed: int = 42) -> float:
    """标签置换检验：评估 AUC 是否优于随机。"""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return np.nan
    obs = roc_auc_score(y_true, y_score)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y_true)
        if len(np.unique(y_perm)) < 2:
            continue
        if roc_auc_score(y_perm, y_score) >= obs:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def delong_auc_test(y_true, y_score_a, y_score_b, n_boot: int = 500, seed: int = 42):
    """Bootstrap ΔAUC 近似检验。"""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    idx = np.arange(len(y_true))
    deltas = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2:
            continue
        auc_a = roc_auc_score(y_true[b], y_score_a[b])
        auc_b = roc_auc_score(y_true[b], y_score_b[b])
        deltas.append(auc_b - auc_a)
    if not deltas:
        return np.nan, np.nan, np.nan
    delta = np.mean(deltas)
    p = 2 * min(np.mean(np.array(deltas) <= 0), np.mean(np.array(deltas) >= 0))
    return delta, p, (np.quantile(deltas, 0.025), np.quantile(deltas, 0.975))


def _gower_center(distance_matrix: np.ndarray) -> np.ndarray:
    """将距离矩阵转为 Gower 双中心化矩阵（dbRDA / PERMANOVA 用）。"""
    n = distance_matrix.shape[0]
    a = -0.5 * (np.asarray(distance_matrix, dtype=float) ** 2)
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ a @ h


def _hat_matrix(x: np.ndarray) -> np.ndarray:
    """投影矩阵 H = X (X'X)^+ X'。"""
    return x @ np.linalg.pinv(x.T @ x) @ x.T


def permanova(distance_matrix: np.ndarray, groups: np.ndarray, permutations: int = 999, seed: int = 42):
    """简化 PERMANOVA 实现。"""
    groups = np.asarray(groups)
    n = distance_matrix.shape[0]
    unique_groups = np.unique(groups)
    ss_total = (distance_matrix ** 2).sum() / n
    ss_within = 0.0
    ss_between = 0.0
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        sub = distance_matrix[np.ix_(idx, idx)]
        ss_within += (sub ** 2).sum() / len(idx)
    ss_between = ss_total - ss_within
    df_between = len(unique_groups) - 1
    df_within = n - len(unique_groups)
    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / max(df_within, 1)
    f_stat = ms_between / ms_within if ms_within > 0 else np.nan
    r2 = ss_between / ss_total if ss_total > 0 else np.nan

    rng = np.random.default_rng(seed)
    perm_f = []
    for _ in range(permutations):
        perm_groups = rng.permutation(groups)
        ss_w = 0.0
        for g in unique_groups:
            idx = np.where(perm_groups == g)[0]
            sub = distance_matrix[np.ix_(idx, idx)]
            ss_w += (sub ** 2).sum() / len(idx)
        ss_b = ss_total - ss_w
        perm_f.append((ss_b / max(df_between, 1)) / (ss_w / max(df_within, 1)))
    p = (np.sum(np.array(perm_f) >= f_stat) + 1) / (permutations + 1)
    return {"F": f_stat, "R2": r2, "p": p}


def permanova_with_covariates(
    distance_matrix: np.ndarray,
    groups: np.ndarray,
    covariates: np.ndarray | pd.DataFrame | None = None,
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """控制协变量后的 PERMANOVA（dbRDA：组效应在协变量残差空间中检验）。

    covariates: shape (n, p)，如 ASA + 麻醉时长；可为 DataFrame。
    """
    groups = np.asarray(groups)
    n = distance_matrix.shape[0]
    g = _gower_center(distance_matrix)

    if covariates is None:
        x_cov = np.ones((n, 1))
    else:
        if isinstance(covariates, pd.DataFrame):
            cov = covariates.to_numpy(dtype=float)
        else:
            cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        x_cov = np.column_stack([np.ones(n), cov])

    # 组编码（二分类或哑变量）
    levels = np.unique(groups)
    if len(levels) < 2:
        return {"F": np.nan, "R2": np.nan, "p": np.nan, "n_covariates": x_cov.shape[1] - 1}

    if len(levels) == 2:
        z = (groups == levels[1]).astype(float).reshape(-1, 1)
        df1 = 1
    else:
        z = pd.get_dummies(groups, drop_first=True).to_numpy(dtype=float)
        df1 = z.shape[1]

    # 残差化：先去除协变量，再检验组效应（Type III / partial）
    h_cov = _hat_matrix(x_cov)
    i_h = np.eye(n) - h_cov
    g_res = i_h @ g @ i_h
    z_res = i_h @ z

    # 若组变量与协变量完全共线，无法检验
    if np.linalg.matrix_rank(z_res, tol=1e-8) < df1:
        return {
            "F": np.nan,
            "R2": np.nan,
            "p": np.nan,
            "n_covariates": int(x_cov.shape[1] - 1),
            "note": "group collinear with covariates",
        }

    h_z = _hat_matrix(z_res)
    ss_group = float(np.trace(h_z @ g_res))
    ss_total = float(np.trace(g_res))
    ss_resid = ss_total - ss_group
    df2 = n - x_cov.shape[1] - df1
    if df2 <= 0 or ss_resid <= 0:
        f_stat = np.nan
    else:
        f_stat = (ss_group / df1) / (ss_resid / df2)
    r2 = ss_group / ss_total if ss_total > 0 else np.nan

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(permutations):
        z_perm = rng.permutation(z)
        z_perm_res = i_h @ z_perm
        if np.linalg.matrix_rank(z_perm_res, tol=1e-8) < df1:
            continue
        h_perm = _hat_matrix(z_perm_res)
        ss_g = float(np.trace(h_perm @ g_res))
        ss_r = ss_total - ss_g
        if ss_r <= 0:
            continue
        f_perm = (ss_g / df1) / (ss_r / df2)
        if f_perm >= f_stat:
            ge += 1
    p = (ge + 1) / (permutations + 1)
    return {
        "F": float(f_stat) if pd.notna(f_stat) else np.nan,
        "R2": float(r2) if pd.notna(r2) else np.nan,
        "p": float(p),
        "n_covariates": int(x_cov.shape[1] - 1),
        "covariates": "yes" if covariates is not None else "no",
    }


def clr_transform(abundance: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    mat = abundance.astype(float) + pseudocount
    log_mat = np.log(mat)
    geom = log_mat.mean(axis=1)
    return log_mat.sub(geom, axis=0)


def cliffs_delta(x: pd.Series, y: pd.Series) -> float:
    """Cliff's delta 效应量（Mann-Whitney 配套）。"""
    x = pd.to_numeric(x, errors="coerce").dropna().values
    y = pd.to_numeric(y, errors="coerce").dropna().values
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = sum(a > b for a in x for b in y)
    lt = sum(a < b for a in x for b in y)
    return float((gt - lt) / (len(x) * len(y)))


def betadisper_test(distance_matrix: np.ndarray, groups: np.ndarray) -> dict:
    """组内离散度（betadisper 简化版）：比较各样本到组内均距，Mann-Whitney。"""
    groups = np.asarray(groups)
    dispersions: dict[str, list[float]] = {}
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        sub = distance_matrix[np.ix_(idx, idx)]
        dispersions[str(g)] = sub.mean(axis=1).tolist()
    keys = list(dispersions.keys())
    if len(keys) != 2:
        return {"p": float("nan"), "statistic": float("nan"), "groups": keys}
    a, b = dispersions[keys[0]], dispersions[keys[1]]
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"p": float(p), "statistic": float(stat), "groups": keys}


def bootstrap_delta_auc_ci(
    y_true, y_score_a, y_score_b, n_boot: int = 1000, ci: float = 0.95, seed: int = 42
):
    """Bootstrap ΔAUC = AUC(B) - AUC(A) 置信区间。"""
    delta, p, (low, high) = delong_auc_test(y_true, y_score_a, y_score_b, n_boot=n_boot, seed=seed)
    return float(delta), float(low), float(high), float(p)


def format_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
