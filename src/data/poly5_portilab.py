"""
Read TMSi / Portilab Poly5 waveform files (.S00, .poly5, etc.).

Binary layout matches SWELL's ``tms_read.m``
(``0 - Raw data/.../Matlab script to read s00 files/tms_read.m``).
Output is **microvolts (uV)** per channel, same affine calibration as MATLAB.

Large files: reads block-by-block into a preallocated ``float32`` array (no giant Python lists).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Poly5ReadResult:
    path: Path
    version: int
    sample_rate_hz: float
    measurement_name: str
    channel_names: list[str]
    """Logical channel names (``NS/2`` in TMSi terms)."""
    data_uv: np.ndarray
    """Shape ``(C, T)``, ``float32``, microvolts."""
    num_sample_blocks: int
    size_signal_data_block: int
    number_of_signals: int


def _parse_header(fid: bytes) -> tuple[int, float, str, int, int, int, int, list[dict], int]:
    """Return version, fs, measurement_name, ns, nb, sd, ns_32bit, descriptions, first_data_block_offset."""
    version = struct.unpack_from("<h", fid, 32)[0]
    if version not in (203, 204):
        raise ValueError(f"Unsupported Poly5 version {version} (expected 203 or 204)")
    if version == 203:
        start_descriptions = 217
        off = 33  # 31-byte FID + int16 version
    else:
        start_descriptions = 218
        off = 34  # 32-byte FID + int16 version

    meas = fid[off : off + 81]
    off += 81
    meas_len = meas[0]
    measurement_name = meas[1 : 1 + meas_len].decode("latin-1", errors="replace")

    fs, storage_rate, storage_type, ns = struct.unpack_from("<h h B h", fid, off)
    off += 2 + 2 + 1 + 2
    _n_sample_periods = struct.unpack_from("<i", fid, off)[0]
    off += 4
    off += 4  # EMPTYBYTES
    off += 14  # StartMeasurement
    nb = struct.unpack_from("<i", fid, off)[0]
    off += 4
    _spp, sd, _delta = struct.unpack_from("<H H h", fid, off)
    off += 2 + 2 + 2
    off += 64  # TrailingZeros

    if off != start_descriptions:
        raise ValueError(f"header field layout mismatch: off={off} expected={start_descriptions}")

    descriptions: list[dict] = []
    for _ in range(ns):
        sn = fid[off : off + 41]
        off += 41
        off += 4
        un = fid[off : off + 11]
        off += 11
        unit_lo, unit_hi, adc_lo, adc_hi, _idx, _cache = struct.unpack_from("<4f h h", fid, off)
        off += 4 * 4 + 2 + 2
        off += 60
        sn_len = sn[0]
        signal_name = sn[1 : 1 + sn_len].decode("latin-1", errors="replace")
        un_len = un[0]
        unit_name = un[1 : 1 + un_len].decode("latin-1", errors="replace")
        descriptions.append(
            {
                "signal_name": signal_name,
                "unit_name": unit_name,
                "unit_low": unit_lo,
                "unit_high": unit_hi,
                "adc_low": adc_lo,
                "adc_high": adc_hi,
            }
        )

    if ns % 2 != 0:
        raise ValueError(f"NumberOfSignals={ns} is odd; expected even (paired TMSi channels).")

    ns_32bit = ns // 2
    return version, float(fs), measurement_name, ns, nb, sd, ns_32bit, descriptions, off


def read_poly5(
    path: str | Path,
    *,
    max_blocks: int | None = None,
    drop_empty_channels: bool = True,
) -> Poly5ReadResult:
    """
    Read a Poly5 / Portilab file into ``(C, T)`` microvolts.

    Parameters
    ----------
    path
        Path to ``.S00`` (or other Poly5) file.
    max_blocks
        If set, only read the first ``max_blocks`` sample blocks (for quick inspection).
    drop_empty_channels
        Drop channels with mean 0 and std 0 (same idea as MATLAB ``remove_not_measured_chan``).
    """
    path = Path(path)
    with path.open("rb") as fp:
        head = fp.read(50_000)
    if len(head) < 5000:
        raise FileNotFoundError(path)

    version, fs, measurement_name, ns, nb, sd, ns_32bit, descriptions, pos0 = _parse_header(head)

    n_float_block = sd // 4
    ns_32bit_chk = ns // 2
    if n_float_block % ns_32bit_chk != 0:
        raise ValueError(f"Inconsistent block: SD={sd}, NS/2={ns_32bit_chk}")
    n_cols = n_float_block // ns_32bit_chk
    nb_use = nb if max_blocks is None else min(nb, max_blocks)
    T = n_cols * nb_use

    out = np.zeros((ns_32bit, T), dtype=np.float32)

    block_header = 4 + 4 + 14 + 64
    with path.open("rb") as fp:
        for g in range(nb_use):
            pos = pos0 + g * (86 + sd) + block_header
            fp.seek(pos)
            raw = fp.read(sd)
            if len(raw) != sd:
                raise ValueError(f"Short read at block {g}: got {len(raw)} expected {sd}")
            block = np.frombuffer(raw, dtype=np.float32).reshape((ns_32bit, n_cols), order="F")
            out[:, g * n_cols : (g + 1) * n_cols] = block

    # uV calibration (MATLAB loop over g=1:NS_32bit uses description(g*2))
    for c in range(ns_32bit):
        desc = descriptions[2 * (c + 1) - 1]
        adc_lo = np.float32(desc["adc_low"])
        adc_hi = np.float32(desc["adc_high"])
        u_lo = np.float32(desc["unit_low"])
        u_hi = np.float32(desc["unit_high"])
        denom = adc_hi - adc_lo
        if denom == 0.0:
            denom = np.float32(1.0)
        scale = (u_hi - u_lo) / denom
        out[c, :] = (out[c, :] - adc_lo) * scale + u_lo

    names = [descriptions[2 * (i + 1) - 1]["signal_name"] for i in range(ns_32bit)]

    if drop_empty_channels:
        keep = []
        for c in range(ns_32bit):
            col = out[c]
            if float(col.mean()) == 0.0 and float(col.std()) == 0.0:
                continue
            keep.append(c)
        out = out[keep, :].copy()
        names = [names[i] for i in keep]

    return Poly5ReadResult(
        path=path,
        version=version,
        sample_rate_hz=fs,
        measurement_name=measurement_name,
        channel_names=names,
        data_uv=out,
        num_sample_blocks=int(nb_use),
        size_signal_data_block=int(sd),
        number_of_signals=int(ns),
    )


def poly5_to_numpy(result: Poly5ReadResult) -> tuple[float, list[str], np.ndarray]:
    """Return ``fs, channel_names, array (C, T)`` (same as ``read_poly5`` array)."""
    return result.sample_rate_hz, result.channel_names, result.data_uv
