from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from phase.data.manifest import AUDIO_SUFFIXES, resolve

BLOCK = 1 << 20

HEADROOM = 0.95

DEAD_RMS = 1e-5

LOW_FREQ_HZ = 10.0

LOW_FREQ_FRACTION = 0.9


def measure(path: Path) -> dict[str, Any]:
    total = 0
    accum = 0.0
    peak = 0.0
    with sf.SoundFile(str(path)) as handle:
        sample_rate = handle.samplerate
        for block in handle.blocks(blocksize=BLOCK, dtype="float64", always_2d=False):
            if block.ndim > 1:
                block = block.mean(axis=1)
            accum += float(block.sum())
            total += block.size
            peak = max(peak, float(np.abs(block).max()))
    dc = accum / total if total else 0.0

    energy = 0.0
    ac_peak = 0.0
    with sf.SoundFile(str(path)) as handle:
        for block in handle.blocks(blocksize=BLOCK, dtype="float64", always_2d=False):
            if block.ndim > 1:
                block = block.mean(axis=1)
            centred = block - dc
            energy += float((centred**2).sum())
            ac_peak = max(ac_peak, float(np.abs(centred).max()))

    return {
        "dc": dc,
        "peak": peak,
        "ac_peak": ac_peak,
        "ac_rms": float(np.sqrt(energy / total)) if total else 0.0,
        "samples": total,
        "sample_rate": sample_rate,
    }


def low_frequency_fraction(path: Path, dc: float, seconds: float = 20.0) -> float:
    with sf.SoundFile(str(path)) as handle:
        frames = min(handle.frames, int(seconds * handle.samplerate))
        data = handle.read(frames=frames, dtype="float64", always_2d=False)
        sample_rate = handle.samplerate
    if data.ndim > 1:
        data = data.mean(axis=1)
    centred = data - dc
    if centred.size < 16 or not np.any(centred):
        return 0.0
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(centred.size))) ** 2
    freqs = np.fft.rfftfreq(centred.size, 1 / sample_rate)
    total = spectrum.sum()
    return float(spectrum[freqs < LOW_FREQ_HZ].sum() / total) if total else 0.0


def survey(root: Path) -> dict[str, Any]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise SystemExit(f"no audio under {root}")

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(files, 1):
        stats = measure(path)
        stats["path"] = path.relative_to(root).as_posix()
        stats["low_freq_fraction"] = low_frequency_fraction(path, stats["dc"])
        rows.append(stats)
        if index % 50 == 0:
            print(f"  measured {index}/{len(files)}", file=sys.stderr)

    global_ac_peak = max(r["ac_peak"] for r in rows)
    gain = HEADROOM / global_ac_peak if global_ac_peak > 0 else 1.0

    flags = {
        "dead_channel": [r["path"] for r in rows if r["ac_rms"] < DEAD_RMS],
        "low_frequency_dominated": [
            r["path"] for r in rows if r["low_freq_fraction"] > LOW_FREQ_FRACTION
        ],
        "negative_dc": [r["path"] for r in rows if r["dc"] < 0],
    }

    dcs = np.array([r["dc"] for r in rows])
    rmss = np.array([r["ac_rms"] for r in rows])
    return {
        "root": root.as_posix(),
        "n_files": len(rows),
        "global_ac_peak": global_ac_peak,
        "global_gain": gain,
        "gain_mode": "per_file",
        "headroom": HEADROOM,
        "dc": {
            "min": float(dcs.min()),
            "median": float(np.median(dcs)),
            "max": float(dcs.max()),
            "max_abs": float(np.abs(dcs).max()),
        },
        "ac_rms": {
            "min": float(rmss.min()),
            "median": float(np.median(rmss)),
            "max": float(rmss.max()),
        },
        "flags": {k: {"count": len(v), "paths": v[:20]} for k, v in flags.items()},
        "rows": rows,
    }


def gain_for(stats: dict[str, Any], report: dict[str, Any]) -> float:
    if report.get("gain_mode") == "global":
        return report["global_gain"]
    peak = stats["ac_peak"]
    return HEADROOM / peak if peak > 0 else 1.0


def apply(report: dict[str, Any], root: Path, dest: Path) -> dict[str, Any]:
    by_path = {r["path"]: r for r in report["rows"]}
    written = 0

    for relative, stats in sorted(by_path.items()):
        gain = gain_for(stats, report)
        stats["gain"] = gain
        source = root / relative
        target = (dest / relative).with_suffix(".flac")
        if target.is_file():
            written += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)

        with (
            sf.SoundFile(str(source)) as handle,
            sf.SoundFile(
                str(target),
                mode="w",
                samplerate=handle.samplerate,
                channels=1,
                format="FLAC",
                subtype="PCM_16",
            ) as out,
        ):
            for block in handle.blocks(blocksize=BLOCK, dtype="float64", always_2d=False):
                if block.ndim > 1:
                    block = block.mean(axis=1)
                out.write(np.clip((block - stats["dc"]) * gain, -1.0, 1.0))
        written += 1
        if written % 50 == 0:
            print(f"  wrote {written}/{len(by_path)}", file=sys.stderr)

    return {"written": written, "dest": dest.as_posix()}


def verify(report: dict[str, Any], root: Path, dest: Path, sample: int = 12) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    rows = report["rows"]
    picks = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)

    residual_dc = []
    max_error = 0.0
    for index in picks:
        stats = rows[int(index)]
        gain = gain_for(stats, report)
        source = root / stats["path"]
        target = (dest / stats["path"]).with_suffix(".flac")
        frames = min(stats["samples"], 32000 * 20)

        raw, _ = sf.read(str(source), frames=frames, dtype="float64", always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        expected = np.clip((raw - stats["dc"]) * gain, -1.0, 1.0)
        expected = np.round(expected * 32768).astype(np.int32)

        got, _ = sf.read(str(target), frames=frames, dtype="int16", always_2d=False)
        max_error = max(
            max_error, float(np.abs(expected - got.astype(np.int32)).max())
        )
        residual_dc.append(float(got.astype(np.float64).mean() / 32768))

    return {
        "checked": len(picks),
        "max_int16_error": max_error,
        "max_residual_dc": float(np.abs(residual_dc).max()),
    }


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in report.items() if k != "rows"}
    slim["rows"] = report["rows"]
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return out


def render(report: dict[str, Any], applied: dict[str, Any], checked: dict[str, Any]) -> str:
    gains = [r.get("gain", 1.0) for r in report["rows"]]
    bits = (
        np.log2(np.maximum([r["ac_peak"] * g for r, g in zip(report["rows"], gains)], 1e-12)) + 15
    )
    lines = [
        f"files            {report['n_files']}",
        f"gain mode        {report.get('gain_mode', 'global')} (headroom {report['headroom']})",
        f"global AC peak   {report['global_ac_peak']:.6f}",
        f"gain             min {min(gains):.3f}  median {float(np.median(gains)):.3f}  max {max(gains):.3f}",
        f"effective bits   min {bits.min():.1f}  p05 {float(np.percentile(bits, 5)):.1f}  median {float(np.median(bits)):.1f}",
        f"DC before        min {report['dc']['min']:+.5f}  median {report['dc']['median']:+.5f}  "
        f"max {report['dc']['max']:+.5f}",
        f"AC rms           min {report['ac_rms']['min']:.5f}  median {report['ac_rms']['median']:.5f}  "
        f"max {report['ac_rms']['max']:.5f}",
        "",
        "flags",
    ]
    for name, block in report["flags"].items():
        lines.append(f"  {name:26s} {block['count']:>4d}")
        for path in block["paths"][:3]:
            lines.append(f"      {path}")
    lines += [
        "",
        f"written          {applied['written']} FLAC files to {applied['dest']}",
        f"round trip       {checked['checked']} files, max int16 error {checked['max_int16_error']:.0f}",
        f"residual DC      {checked['max_residual_dc']:.2e}",
    ]
    return "\n".join(lines)
