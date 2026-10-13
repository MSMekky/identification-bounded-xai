"""Physical mechanism for ir_3, fitted on randomized rows.

ir_3 = (a . RGB) * (cos^2(pol_1 - pol_2) + g) + e0 + e1 * l_31 + e2 * l_32

This is Malus' law for two imperfect polarizers times the source intensity seen by the infrared
channel, plus the small effect of the LEDs beside the third sensor (Gamella et al. 2025, App. IV.2.1).
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def malus(pol1, pol2):
    return np.cos(np.deg2rad(np.asarray(pol1) - np.asarray(pol2))) ** 2


@dataclass
class Mechanism:
    params: np.ndarray
    r2: float
    proxy: np.ndarray          # linear map RGB -> ir_1 on unsaturated rows: (w_r, w_g, w_b, w_0)
    proxy_r2: float

    def intensity(self, df):
        return df[["red", "green", "blue"]].to_numpy(float) @ self.params[:3]

    def ir3(self, intensity, pol1, pol2, l31, l32):
        p = self.params
        return intensity * (malus(pol1, pol2) + p[3]) + p[4] + p[5] * l31 + p[6] * l32

    def predict(self, df):
        return self.ir3(self.intensity(df), df.pol_1, df.pol_2, df.l_31, df.l_32)

    def intensity_from_proxy(self, ir1):
        """Invert the ir_1 map onto the ir_3 intensity scale (saturated readings stay censored)."""
        w = self.proxy
        k = np.dot(w[:3], self.params[:3]) / np.dot(self.params[:3], self.params[:3])
        return (np.asarray(ir1) - w[3]) / k


def fit(df):
    X = df[["red", "green", "blue"]].to_numpy(float)
    c2, y = malus(df.pol_1, df.pol_2), df.ir_3.to_numpy(float)
    L = df[["l_31", "l_32"]].to_numpy(float)

    def f(p):
        return (X @ p[:3]) * (c2 + p[3]) + p[4] + L @ p[5:7]

    r = least_squares(lambda p: f(p) - y, np.array([2.0, 1.5, 1.0, 0.03, 0.0, 0.0, 0.0]))
    r2 = 1 - np.var(y - f(r.x)) / np.var(y)
    ok = df.ir_1.to_numpy() < 65000
    A = np.c_[X, np.ones(len(X))]
    w = np.linalg.lstsq(A[ok], df.ir_1.to_numpy(float)[ok], rcond=None)[0]
    pr2 = 1 - np.var(df.ir_1[ok] - A[ok] @ w) / np.var(df.ir_1[ok])
    return Mechanism(r.x, float(r2), w, float(pr2))
