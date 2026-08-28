from __future__ import annotations

import argparse

from phase.config import REPO_ROOT
from phase.data import clean
from phase.data.manifest import resolve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DC-remove and re-encode a corpus to 16-bit FLAC")
    parser.add_argument("--root", default="data/raw/deepship/deepship-raw")
    parser.add_argument("--dest", default="data/clean/deepship")
    parser.add_argument("--report", default="data/manifests/deepship_clean_report.json")
    parser.add_argument("--survey-only", action="store_true")
    args = parser.parse_args(argv)

    root = resolve(args.root)
    dest = resolve(args.dest)

    print(f"surveying {root}")
    report = clean.survey(root)

    if args.survey_only:
        out = clean.write_report(report, args.report)
        print(
            clean.render(
                report,
                {"written": 0, "dest": str(dest)},
                {"checked": 0, "max_int16_error": 0.0, "max_residual_dc": 0.0},
            )
        )
        print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}")
        return 0

    print(f"writing {dest}")
    applied = clean.apply(report, root, dest)
    checked = clean.verify(report, root, dest)

    out = clean.write_report(report, args.report)
    print(clean.render(report, applied, checked))
    print(f"wrote {out.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
