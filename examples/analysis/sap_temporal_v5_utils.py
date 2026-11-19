"""CPU-only mathematical diagnostics for SAP's temporal-action evidence (V5).

These helpers intentionally have no LeRobot, Torch, or GPU dependency so the
order-information null and synthetic AR(2) identification can be unit tested.
"""

from __future__ import annotations

import math

import numpy as np


def parse_names(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names or len(names) != len(set(names)) or any(name not in allowed for name in names):
        raise ValueError(f"Expected distinct names from {allowed}, got {value!r}")
    return names


def temporal_mask(batch: int, horizon: int, mode: str, rng: np.random.Generator):
    """Return hidden positions and scoring positions, which may be a subset.

    Future forecasting *always hides the entire future*, even when scoring only
    5/10/20 steps; otherwise future ground-truth actions leak through context.
    """
    if horizon != 50 or batch < 1:
        raise ValueError("V5 uses nonempty batches of original PI0.5 50-step chunks")
    hidden = np.zeros((batch, horizon), dtype=bool)
    scored = np.zeros_like(hidden)
    if mode.startswith("future"):
        steps = int(mode[len("future"):])
        if steps not in (5, 10, 20):
            raise ValueError(f"Unsupported forecast horizon: {mode}")
        hidden[:, 20:] = True
        scored[:, 20:20 + steps] = True
    elif mode in ("middle10", "middle20"):
        steps = int(mode[len("middle"):])
        start = (horizon - steps) // 2
        hidden[:, start:start + steps] = True
        scored[:] = hidden
    elif mode == "random20":
        for row in hidden:
            row[rng.choice(horizon, size=20, replace=False)] = True
        scored[:] = hidden
    else:
        raise ValueError(f"Unknown temporal-mask mode: {mode}")
    if np.any(scored & ~hidden) or np.any(hidden.all(axis=1)):
        raise AssertionError("Invalid target mask: leakage or no visible context")
    return hidden, scored


MASK_MODES = ("future5", "future10", "future20", "middle10", "middle20", "random20")


def swap_visible_order(actions: np.ndarray, hidden: np.ndarray, rng: np.random.Generator):
    """Permute whole *visible action vectors*, leaving every target untouched."""
    result = np.array(actions, copy=True)
    if result.ndim != 3 or hidden.shape != result.shape[:2]:
        raise ValueError("Expected actions [B,T,D] and hidden mask [B,T]")
    for batch_index in range(len(result)):
        visible = np.flatnonzero(~hidden[batch_index])
        if len(visible) > 1:
            result[batch_index, visible] = actions[batch_index, rng.permutation(visible)]
    return result


def ordered_history(chunks: np.ndarray, context: int = 20):
    return np.asarray(chunks[:, :context], dtype=np.float64).reshape(len(chunks), -1)


def unordered_history(chunks: np.ndarray, context: int = 20):
    """Lexicographically canonicalize *whole vectors*; preserve set and shape.

    Per-channel sorting would destroy simultaneous inter-joint relationships,
    making the comparison unfair. The ordered and bag predictors have exactly
    the same input dimensionality and regression family.
    """
    prefix = np.asarray(chunks[:, :context], dtype=np.float64)
    result = np.empty_like(prefix)
    for index, vectors in enumerate(prefix):
        # Exact ties mean identical full vectors and cannot encode the order.
        # Sorting solely by norm would leak order through equal-norm ties.
        keys = tuple(vectors[:, channel] for channel in reversed(range(vectors.shape[1])))
        result[index] = vectors[np.lexsort(keys)]
    return result.reshape(len(prefix), -1)


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray,
                   x_eval: np.ndarray, alpha: float) -> np.ndarray:
    """Dual ridge regression, with intercept left unregularized."""
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    centered = x_train - x_mean
    centered_y = y_train - y_mean
    kernel = centered @ centered.T
    scale = max(float(np.trace(kernel) / len(kernel)), 1e-8)
    weights = np.linalg.solve(kernel + alpha * scale * np.eye(len(kernel)), centered_y)
    return (x_eval - x_mean) @ centered.T @ weights + y_mean


def ridge_forecast(train: np.ndarray, test: np.ndarray, *,
                   context: int = 20, horizon: int = 10, seed: int = 117):
    """Capacity-matched ordered-vs-bag held-out future prediction probe.

    The theorem is about Bayes risks; a restricted ridge predictor *only*
    diagnoses accessible order information and does not estimate that identity.
    """
    if not 2 <= context < train.shape[1] or not 1 <= horizon <= train.shape[1] - context:
        raise ValueError("Invalid context or future horizon")
    if len(train) < 12 or len(test) < 2:
        raise ValueError("Need at least 12 train chunks and 2 held-out test chunks")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train))
    validation_count = max(2, int(round(len(indices) * .2)))
    fit_ids, validation_ids = indices[validation_count:], indices[:validation_count]
    y_train = np.asarray(train[:, context:context + horizon], dtype=np.float64).reshape(len(train), -1)
    y_test = np.asarray(test[:, context:context + horizon], dtype=np.float64).reshape(len(test), -1)
    baseline = np.repeat(y_train[fit_ids].mean(axis=0, keepdims=True), len(test), axis=0)
    denominator = max(float(np.mean((y_test - baseline) ** 2)), 1e-12)
    outcomes = {}
    for name, encoder in (("ordered", ordered_history), ("unordered", unordered_history)):
        features = encoder(train, context=context)
        test_features = encoder(test, context=context)
        best = None
        for alpha in (.001, .01, .1, 1., 10., 100.):
            prediction = _ridge_predict(features[fit_ids], y_train[fit_ids],
                                        features[validation_ids], alpha)
            loss = float(np.mean((prediction - y_train[validation_ids]) ** 2))
            if best is None or loss < best[0]:
                best = (loss, alpha)
        forecast = _ridge_predict(features[fit_ids], y_train[fit_ids], test_features, best[1])
        errors = np.mean((forecast - y_test) ** 2, axis=1)
        outcomes[name] = {"relative_mse": float(np.mean(errors) / denominator),
                          "ridge_alpha": best[1], "per_episode_mse": errors.tolist()}
    outcomes["intercept"] = {"relative_mse": float(np.mean((baseline - y_test) ** 2) / denominator)}
    return outcomes


def paired_interval(better: np.ndarray, worse: np.ndarray, *,
                    seed: int, iterations: int = 1000):
    """Return the paired improvement worse-better, resampling *episodes*."""
    a, b = np.asarray(better, dtype=np.float64), np.asarray(worse, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite(a + b).all():
        raise ValueError("Need finite paired episode-level score vectors")
    difference = b - a
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(iterations, len(a)))
    intervals = np.quantile(difference[indices].mean(axis=1), (.025, .975))
    return {"episodes": len(a), "mean": float(difference.mean()),
            "ci95": [float(value) for value in intervals]}


def synthetic_ar2_chunks(n: int, *, horizon: int = 50, dim: int = 7,
                         seed: int = 0, first_episode: int = 0):
    """Known stable AR(2) temporal-action dynamics, with three modal regimes."""
    if n < 2 or horizon < 15 or dim < 3:
        raise ValueError("Need >=2 examples, >=15 timesteps, >=3 action channels")
    rng = np.random.default_rng(seed)
    regimes = ((.94, .21 * math.pi), (.88, .12 * math.pi), (.96, .36 * math.pi))
    result = np.empty((n, horizon, dim), dtype=np.float32)
    metadata = []
    for i in range(n):
        regime = i % len(regimes)
        r, omega = regimes[regime]
        b1, b2 = 2.0 * r * math.cos(omega), -(r ** 2)
        x = np.empty((horizon, dim - 1), dtype=np.float64)
        x[0] = rng.normal(0., .22, size=dim - 1)
        x[1] = rng.normal(0., .22, size=dim - 1)
        for timestep in range(2, horizon):
            x[timestep] = b1 * x[timestep - 1] + b2 * x[timestep - 2]
            x[timestep] += rng.normal(0., .012, size=dim - 1)
        scale = max(1., float(np.max(np.abs(x)) / .8))
        x /= scale  # A scalar change of units preserves the AR(2) coefficients.
        result[i, :, :-1] = x.astype(np.float32)
        result[i, :, -1] = np.where(x[:, 0] > 0., .6, -.6)
        metadata.append({"episode_index": first_episode + i, "offset": 0,
                         "episode_length": horizon, "task": f"ar2-regime-{regime}",
                         "stratum": "transition", "regime": regime,
                         "true_ar2": {"r": r, "omega": omega,
                                      "beta1": b1, "beta2": b2}})
    return result, metadata


def ar2_parameter_error(path: np.ndarray, *, beta1: float, beta2: float,
                        start: int = 20, stop: int = 40):
    """Fit AR(2) coefficients on the *generated* hidden region only."""
    x = np.asarray(path, dtype=np.float64)
    if x.ndim != 2 or stop > len(x) or stop - start < 6 or x.shape[1] < 3:
        raise ValueError("Invalid sequence or hidden region")
    continuous = x[start:stop, :-1]
    lhs = np.stack((continuous[1:-1], continuous[:-2]), axis=-1).reshape(-1, 2)
    rhs = continuous[2:].reshape(-1)
    coefficients, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    if rank < 2:
        return {"coefficient_l2": None, "beta1_hat": None, "beta2_hat": None,
                "r_hat": None, "omega_hat": None}
    # Oscillatory complex poles exist only when the fitted AR(2) polynomial
    # has negative beta2 and |beta1/(2r)| <= 1. Do not clip an invalid model
    # to a fake oscillatory solution; report the pole estimate as undefined.
    r_hat = math.sqrt(-float(coefficients[1])) if coefficients[1] < 0 else None
    cos_omega = float(coefficients[0]) / (2. * r_hat) if r_hat and r_hat > 1e-12 else None
    omega_hat = (math.acos(cos_omega) if cos_omega is not None and abs(cos_omega) <= 1.
                 else None)
    return {"coefficient_l2": float(np.linalg.norm(coefficients - [beta1, beta2])),
            "beta1_hat": float(coefficients[0]), "beta2_hat": float(coefficients[1]),
            "r_hat": r_hat, "omega_hat": omega_hat}


def signature_features(paths: np.ndarray, *, level: int = 2):
    """Time-augmented truncated path signatures (levels 1 and 2).

    The finite truncation is NOT the characteristic robust signature kernel.
    We use it as a transparent, order-sensitive finite-sample diagnostic.
    """
    if level != 2 or paths.ndim != 3 or paths.shape[1] < 2:
        raise ValueError("Expected [N,T,D] paths and a level-2 signature")
    n, horizon, dims = paths.shape
    time = np.broadcast_to(np.linspace(0., 1., horizon)[None, :, None], (n, horizon, 1))
    increments = np.diff(np.concatenate([paths.astype(np.float64), time], axis=2), axis=1)
    width = dims + 1
    first = np.zeros((n, width), dtype=np.float64)
    second = np.zeros((n, width, width), dtype=np.float64)
    for step in range(horizon - 1):
        delta = increments[:, step]
        second += first[:, :, None] * delta[:, None, :]
        second += .5 * delta[:, :, None] * delta[:, None, :]
        first += delta
    return np.concatenate([first, second.reshape(n, -1)], axis=1)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float):
    squared = np.maximum(np.sum(x * x, axis=1)[:, None]
                         + np.sum(y * y, axis=1)[None, :] - 2. * x @ y.T, 0.)
    return np.exp(-squared / (2. * sigma * sigma))


def _median_bandwidth(x: np.ndarray):
    distances = np.maximum(np.sum(x * x, axis=1)[:, None]
                           + np.sum(x * x, axis=1)[None, :] - 2. * x @ x.T, 0.)
    positive = distances[np.triu_indices(len(x), 1)]
    positive = positive[positive > 1e-12]
    return math.sqrt(float(np.median(positive))) if len(positive) else 1.


def conditional_signature_mmd(true_paths: np.ndarray, predicted_paths: np.ndarray,
                              observed_context: np.ndarray):
    """Biased joint MMD with product kernels k(context)*k(signature).

    With a shared context marginal, the joint-law discrepancy is a proxy for
    context-dependent trajectory modeling. Finite signature order/finite data
    preclude claims of a characteristic conditional metric.
    """
    if true_paths.shape != predicted_paths.shape or len(true_paths) != len(observed_context):
        raise ValueError("The paired real/generated paths must share context and shape")
    if len(true_paths) < 2:
        raise ValueError("At least two independent episodes are required")
    source = signature_features(true_paths)
    target = signature_features(predicted_paths)
    # Scale using the reference distribution ONLY, shared across model variants.
    scale = np.maximum(source.std(axis=0, ddof=1), .05)
    source, target = source / scale, target / scale
    source_bw = _median_bandwidth(source)
    condition = observed_context.reshape(len(observed_context), -1).astype(np.float64)
    condition = condition / np.maximum(condition.std(axis=0, ddof=1), .05)
    condition_kernel = _rbf_kernel(condition, condition, _median_bandwidth(condition))
    real_kernel = _rbf_kernel(source, source, source_bw)
    fake_kernel = _rbf_kernel(target, target, source_bw)
    cross_kernel = _rbf_kernel(source, target, source_bw)
    mmd2 = ((real_kernel * condition_kernel).mean()
            + (fake_kernel * condition_kernel).mean()
            - 2. * (cross_kernel * condition_kernel).mean())
    return max(0., float(mmd2))
