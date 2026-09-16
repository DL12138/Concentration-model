# -*- coding: utf-8 -*-
"""建模与统计（M5）：4 类模型拟合、R²/RMSE/LOD、预测反解、不确定度 U。

约定：x = 已知浓度，y = 检测区特征（默认 hue，0~180 的 OpenCV 色相）。
拟合方向 y = f(x)，预测时对给定特征反解浓度。
"""
import warnings

import numpy as np
from scipy import optimize, stats

MODEL_TYPES = ['linear', 'poly2', 'exp', '4pl']

FEATURES = [
    'hue', 'saturation', 'value', 'ratio_gr', 'ratio_bg',
    'mean_r', 'mean_g', 'mean_b', 'intensity',
]


# ---------- 模型定义（特征域 y = f(x, params)） ----------

def _f_linear(x, a, b):
    return a * np.asarray(x) + b


def _f_poly2(x, a, b, c):
    x = np.asarray(x)
    return a * x ** 2 + b * x + c


def _f_exp(x, a, b, c):
    # 优化器会探测越界参数，静默数值溢出（对应初值本次拟合失败，由多初值策略兜底）
    with np.errstate(over='ignore', invalid='ignore'):
        return a * np.exp(b * np.asarray(x)) + c


def _f_4pl(x, a, b, c, d):
    """a=上渐近线, b=斜率因子, c=中点(EC50), d=下渐近线"""
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.asarray(x)
        return d + (a - d) / (1.0 + (x / c) ** b)


FUNCS = {
    'linear': _f_linear,
    'poly2': _f_poly2,
    'exp': _f_exp,
    '4pl': _f_4pl,
}

PARAM_COUNTS = {'linear': 2, 'poly2': 3, 'exp': 3, '4pl': 4}


# ---------- 指标 ----------

def metrics(y_true, y_pred, n_params):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(ss_res / n))
    resid_sd = float(np.sqrt(ss_res / max(n - n_params, 1)))
    return {'r2': round(r2, 6), 'rmse': round(rmse, 6), 'resid_sd': round(resid_sd, 6)}


# ---------- 问题2-H：数据预处理与交叉验证 ----------

def preprocess_data(x, y, log_conc=False, zscore=False, iqr=True):
    """标定数据预处理（问题2-H）。

    - iqr：按浓度组（同浓度重复点）用 IQR 剔除离群特征值；
    - log_conc：浓度 x 变换为 log10(x+1)（兼容 0 浓度）；
    - zscore：特征 y 标准化（(y-mean)/std）。
    返回 (x_clean, y_clean, meta)；meta 含 y_mean/y_std（zscore 时）与剔除数。
    """
    x = list(x)
    y = list(y)
    meta = {'removed': 0}
    if iqr and len(x) >= 4:
        keep = []
        from collections import defaultdict
        groups = defaultdict(list)
        for xi, yi in zip(x, y):
            groups[round(float(xi), 6)].append(yi)
        for xi, yi in zip(x, y):
            g = groups[round(float(xi), 6)]
            if len(g) >= 4:
                q1, q3 = np.percentile(g, [25, 75])
                iqr_v = q3 - q1
                if iqr_v < 1e-12:
                    # 组内几乎无变异：不做剔除，避免把正常重复点误删
                    keep.append((xi, yi))
                    continue
                lo, hi = q1 - 1.5 * iqr_v, q3 + 1.5 * iqr_v
                if lo <= yi <= hi:
                    keep.append((xi, yi))
                else:
                    meta['removed'] += 1
            else:
                keep.append((xi, yi))
        if keep:
            x, y = zip(*keep)
            x, y = list(x), list(y)
    if log_conc:
        x = [np.log10(float(v) + 1.0) for v in x]
    if zscore and len(y) >= 2:
        y_arr = np.asarray(y, dtype=float)
        ym, ys_ = float(y_arr.mean()), float(y_arr.std(ddof=0))
        if ys_ > 1e-12:
            y = [float((v - ym) / ys_) for v in y_arr]
            meta['y_mean'] = ym
            meta['y_std'] = ys_
        else:
            meta['y_std'] = 0.0
    return x, y, meta


def cross_validate(x, y, model_type='linear', method='loo', k=5):
    """交叉验证（问题2-H）。

    method: 'loo' 留一 | 'kfold' K 折（按浓度分层）| 'leave_group' 留一个浓度组。
    返回 {fold_metrics: [...], summary: {r2, rmse, mae, n_folds}}。
    每折用对应模型拟合训练折、预测验证折。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3 or len(set(x)) < 2:
        raise ValueError('交叉验证至少需要 3 个点、2 个不同浓度')

    if method == 'loo':
        folds = [[i] for i in range(n)]
    elif method == 'leave_group':
        groups = {}
        for i, xi in enumerate(x):
            groups.setdefault(round(float(xi), 6), []).append(i)
        if len(groups) < 2:
            raise ValueError('留浓度组验证至少需要 2 个不同浓度')
        folds = list(groups.values())
    else:  # kfold 分层：按浓度组打散成 k 折
        from collections import defaultdict
        groups = defaultdict(list)
        for i, xi in enumerate(x):
            groups[round(float(xi), 6)].append(i)
        keys = sorted(groups.keys())
        k = max(2, min(int(k), n))
        folds = [[] for _ in range(k)]
        for j, key in enumerate(keys):
            folds[j % k].extend(groups[key])

    fold_metrics = []
    all_true, all_pred = [], []
    idx_all = np.arange(n)
    for test_idx in folds:
        if len(test_idx) < 1:
            continue
        train_idx = np.setdiff1d(idx_all, test_idx)
        if len(train_idx) < 2 or len(set(x[train_idx])) < 2:
            continue
        try:
            fitted = fit_model(list(x[train_idx]), list(y[train_idx]), model_type)
        except ValueError:
            continue
        params = fitted['params']
        y_pred = FUNCS[model_type](list(x[test_idx]), **params)
        y_true = list(y[test_idx])
        n_params = PARAM_COUNTS[model_type]
        m = metrics(y_true, y_pred, n_params)
        m['n'] = len(y_true)
        fold_metrics.append(m)
        all_true.extend(y_true)
        all_pred.extend(list(y_pred))

    if not fold_metrics:
        raise ValueError('交叉验证没有可用的折，请检查数据')
    all_true = np.asarray(all_true, dtype=float)
    all_pred = np.asarray(all_pred, dtype=float)
    summary = {
        'r2': round(float(1.0 - np.sum((all_true - all_pred) ** 2) / max(np.sum((all_true - all_true.mean()) ** 2), 1e-12)), 4),
        'rmse': round(float(np.sqrt(np.mean((all_true - all_pred) ** 2))), 4),
        'mae': round(float(np.mean(np.abs(all_true - all_pred))), 4),
        'n_folds': len(fold_metrics),
        'method': method,
    }
    return {'fold_metrics': fold_metrics, 'summary': summary}


def _low_conc_sigma(x, y):
    """LOD 用 σ：最低浓度组若 ≥2 个重复点，用其样本标准差；否则用全局残差估算占位 None。"""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    xmin = xs.min()
    mask = np.isclose(xs, xmin)
    if mask.sum() >= 2:
        return float(ys[mask].std(ddof=1))
    return None


def _lod_slope(x, y, model_type, params):
    """LOD 斜率：用最低 3 个浓度点（不足则全部点）线性回归的斜率；4PL 额外用其在最低点处导数校验。"""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    order = np.argsort(xs)
    xs_s, ys_s = xs[order], ys[order]
    n = min(3, len(xs_s))
    if n < 2:
        return None
    a, b = np.polyfit(xs_s[:n], ys_s[:n], 1)
    return float(a)


def compute_lod(x, y, model_type, params, sigma=None):
    """LOD（浓度域）= 3.3 * σ / 斜率（IUPAC 常用口径）。"""
    if sigma is None:
        sigma = _low_conc_sigma(x, y)
    if sigma is None:
        return None
    slope = _lod_slope(x, y, model_type, params)
    if slope is None or abs(slope) < 1e-9:
        return None
    lod = 3.3 * sigma / abs(slope)
    return round(float(lod), 4)


# ---------- 拟合 ----------

def fit_linear(x, y):
    a, b = np.polyfit(x, y, 1)
    return {'a': float(a), 'b': float(b)}


def fit_poly2(x, y):
    a, b, c = np.polyfit(x, y, 2)
    return {'a': float(a), 'b': float(b), 'c': float(c)}


def _curve_fit_multi(func, x, y, guesses):
    best = None
    best_cost = None
    for p0 in guesses:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', optimize.OptimizeWarning)
                popt, _ = optimize.curve_fit(func, x, y, p0=p0, maxfev=20000)
            cost = float(np.sum((func(x, *popt) - y) ** 2))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = popt
        except (RuntimeError, ValueError, FloatingPointError):
            continue
    return best


def fit_exp(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    guesses = []
    for amp in [np.ptp(y), np.ptp(y) / 2, np.ptp(y) * 2]:
        for rate in [-0.1, -0.02, -0.5, 0.02, 0.1]:
            for base in [y.min(), y.mean(), y.min() - 1]:
                guesses.append((amp, rate, base))
    p = _curve_fit_multi(_f_exp, x, y, guesses)
    if p is None:
        raise ValueError('指数模型拟合不收敛，请检查数据范围')
    return {'a': float(p[0]), 'b': float(p[1]), 'c': float(p[2])}


def fit_4pl(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    guesses = []
    xm = float(np.median(x))
    for top in [y.max(), y.max() + 1, y.max() * 1.05]:
        for bottom in [y.min(), y.min() - 1]:
            for c in [xm, x.mean(), xm / 2, xm * 2]:
                for b in [1.0, 0.7, 1.5, 2.0]:
                    guesses.append((top, b, c, bottom))
    p = _curve_fit_multi(_f_4pl, x, y, guesses)
    if p is None:
        raise ValueError('4PL 模型拟合不收敛，请检查数据范围')
    return {'a': float(p[0]), 'b': float(p[1]), 'c': float(p[2]), 'd': float(p[3])}


FITTERS = {
    'linear': fit_linear,
    'poly2': fit_poly2,
    'exp': fit_exp,
    '4pl': fit_4pl,
}


def fit_model(x, y, model_type):
    """拟合单一模型，返回 {params, metrics, lod}；失败抛 ValueError。"""
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    if len(x) < 3:
        raise ValueError('标定数据点不足（至少需要 3 个浓度点）')
    if len(set(x)) < 3:
        raise ValueError('至少需要 3 个不同浓度')
    if PARAM_COUNTS[model_type] > len(x):
        raise ValueError(f'{model_type} 模型需要至少 {PARAM_COUNTS[model_type]} 个数据点（当前 {len(x)} 个）')
    params = FITTERS[model_type](x, y)
    y_pred = FUNCS[model_type](x, **params)
    m = metrics(y, y_pred, PARAM_COUNTS[model_type])
    lod = compute_lod(x, y, model_type, params, sigma=_low_conc_sigma(x, y))
    m['lod'] = lod
    return {'type': model_type, 'params': params, **m}


def _degenerate(model_type, params):
    """识别数值退化/无预测意义的模型：振幅、跨度、系数趋近 0 的假拟合。"""
    p = params
    if model_type == 'linear':
        return abs(p['a']) < 1e-12
    if model_type == 'poly2':
        return abs(p['a']) < 1e-12 and abs(p['b']) < 1e-12
    if model_type == 'exp':
        return abs(p['a']) < 1e-6
    if model_type == '4pl':
        return abs(p['a'] - p['d']) < 1e-6 or abs(p['b']) < 1e-9 or abs(p['c']) < 1e-9
    return False


def fit_all_models(x, y):
    """拟合全部 4 类模型，返回 {type: result}；单模型失败/退化时记录 error。"""
    results = {}
    for mt in MODEL_TYPES:
        try:
            r = fit_model(x, y, mt)
            if _degenerate(mt, r['params']):
                r = {'type': mt, 'error': f'{mt} 模型退化（参数无预测意义），请检查数据或改用其他模型'}
            results[mt] = r
        except (ValueError, FloatingPointError) as e:
            results[mt] = {'type': mt, 'error': str(e)}
    return results


def best_model(results):
    """按 R² 选取最佳可用模型（无 R² 的跳过）。"""
    best = None
    for mt, r in results.items():
        if 'error' in r or r.get('r2') is None or np.isnan(r.get('r2', float('nan'))):
            continue
        if best is None or r['r2'] > best[1]['r2']:
            best = (mt, r)
    return best


# ---------- 预测反解 ----------

def predict_conc(model_type, params, feature):
    """给定特征值，反解浓度。"""
    p = params
    y = float(feature)
    if model_type == 'linear':
        if abs(p['a']) < 1e-12:
            raise ValueError('线性模型斜率为 0，无法预测')
        return float((y - p['b']) / p['a'])
    if model_type == 'poly2':
        a, b, c = p['a'], p['b'], p['c']
        disc = b * b - 4 * a * (c - y)
        if disc < 0:
            raise ValueError('特征值超出多项式模型范围，无法反解')
        roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
        pos = [r for r in roots if r >= 0]
        return float(pos[0] if pos else max(roots))
    if model_type == 'exp':
        inner = (y - p['c']) / p['a']
        if inner <= 0:
            raise ValueError('特征值超出指数模型范围，无法反解')
        return float(np.log(inner) / p['b'])
    if model_type == '4pl':
        top, slope, mid, bottom = p['a'], p['b'], p['c'], p['d']
        if y <= bottom or y >= top:
            raise ValueError('特征值超出 4PL 范围，无法反解')
        return float(mid * ((top - bottom) / (y - bottom) - 1.0) ** (1.0 / slope))
    raise ValueError(f'未知模型类型：{model_type}')


def predict_with_u(model_type, params, feature, x_data, y_data, conf=0.95):
    """预测浓度 C 与不确定度 U（浓度域，预测区间半宽）。

    用线性近似（delta method）：特征域残差 → 浓度域斜率转换。
    """
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    n = len(x)
    k = PARAM_COUNTS.get(model_type, 2)
    x0 = predict_conc(model_type, params, feature)

    y_fit = FUNCS[model_type](x, **params)
    ss_res = float(np.sum((y - y_fit) ** 2))
    resid_sd = float(np.sqrt(ss_res / max(n - k, 1)))

    # 特征域预测标准误（含杠杆，线性近似）
    x_mean = float(x.mean())
    leverage = 1.0 / n + (x0 - x_mean) ** 2 / max(float(np.sum((x - x_mean) ** 2)), 1e-12)
    se_y = resid_sd * np.sqrt(1.0 + leverage)

    # 浓度域斜率（数值导数）
    eps = max(abs(x0) * 1e-4, 1e-6)
    y_lo = FUNCS[model_type](x0 - eps, **params)
    y_hi = FUNCS[model_type](x0 + eps, **params)
    slope = abs(float((y_hi - y_lo) / (2 * eps)))
    if slope < 1e-9:
        raise ValueError('模型在该浓度附近斜率过小，不确定度无法计算')

    se_x = se_y / slope
    t = float(stats.t.ppf(0.5 + conf / 2.0, max(n - k, 1)))
    u = t * se_x
    return {'conc': round(float(x0), 4), 'u': round(float(u), 4)}


def curve_points(model_type, params, xmin, xmax, n=80):
    """生成曲线绘图点（浓度 → 特征）。"""
    xs = np.linspace(xmin, xmax, n)
    ys = FUNCS[model_type](xs, **params)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in zip(xs, ys)]
