"""Randomized light-tunnel data: the real Causal Chamber experiment or a calibrated simulation.

Both sources return one DataFrame with the logged actuators (red, green, blue, pol_1, pol_2,
l_31, l_32) and the sensor readings ir_1 and ir_3. Every actuator is drawn independently and
uniformly before each measurement, so the rows form a randomized experiment.
"""
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZIP_URL = "https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/lt_walks_v1.zip"
ZIP_MD5 = "dcad019186661a56de7a2d1db97fc2f0"
CSV_IN_ZIP = "lt_walks_v1/actuators_white.csv"
COLUMNS = ["red", "green", "blue", "pol_1", "pol_2", "l_31", "l_32", "ir_1", "ir_2", "ir_3"]


def load_real(path):
    """Load actuators_white from lt_walks_v1 (a .zip with the published MD5, or the extracted .csv)."""
    path = Path(path)
    if path.suffix == ".zip":
        raw = path.read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        if md5 != ZIP_MD5:
            raise ValueError(f"MD5 mismatch for {path}: {md5} != {ZIP_MD5}")
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            df = pd.read_csv(z.open(CSV_IN_ZIP))
    else:
        df = pd.read_csv(path)
    return df[COLUMNS].astype(float).reset_index(drop=True)


# Calibrated Mk1 sensor model (causalchamber.simulators.lt.Deterministic, tutorial parameters)
_S = np.array([[4.34165202, 2.8594438, 1.91565691], [0.89410569, 1.58679242, 2.68477163]])
_TS = np.array([0.42685688, 0.41231782, 0.46652526])
_TP = np.array([0.2909342, 0.29624314, 0.30526271])
_TC = np.array([0.00431032, 0.01447611, 0.02041075])
_D1, _D2, _D3 = 0.1, 0.17, 0.25
_GAIN_IR = 2 ** (2 + 3)          # diode_ir = 2, t_ir = 3, as in actuators_white


def simulate(n, rng, noise_mult=0.04, noise_add=3.0):
    """Randomized rows from the calibrated simulator with assumed sensor noise.

    Mirrors the actuators_white protocol (ranges and sensor settings). The simulator has no LED
    effects; l_31 and l_32 are drawn but inert.
    """
    rgb = rng.integers(0, 256, (n, 3)).astype(float)
    pol1 = rng.integers(-90, 91, n).astype(float)
    pol2 = rng.integers(-90, 91, n).astype(float)
    malus = np.cos(np.deg2rad(pol1 - pol2)) ** 2
    ir1 = _GAIN_IR * (rgb @ _S[0])
    ir2 = _GAIN_IR * ((_D1 / _D2) ** 2 * rgb * _TS) @ _S[0]
    c = (_D1 / _D3) ** 2 * rgb
    ir3 = _GAIN_IR * ((malus[:, None] * c * (_TP - _TC) + c * _TC) @ _S[0])

    def noisy(x):
        return x * (1 + noise_mult * rng.standard_normal(n)) + noise_add * rng.standard_normal(n)

    return pd.DataFrame(dict(
        red=rgb[:, 0], green=rgb[:, 1], blue=rgb[:, 2], pol_1=pol1, pol_2=pol2,
        l_31=rng.integers(0, 256, n).astype(float), l_32=rng.integers(0, 256, n).astype(float),
        ir_1=np.minimum(noisy(ir1), 65535), ir_2=noisy(ir2), ir_3=noisy(ir3)))
