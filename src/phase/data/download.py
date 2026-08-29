from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

from phase.config import REPO_ROOT, DatasetConfig, load_config

CONFIG_DIR = REPO_ROOT / "configs" / "data"

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aif", ".aiff"}

MANUAL_INSTRUCTIONS: dict[str, str] = {}


def available_datasets() -> dict[str, DatasetConfig]:
    return {
        path.stem: load_config(path, DatasetConfig) for path in sorted(CONFIG_DIR.glob("*.yaml"))
    }


def count_audio(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


def _resolve(root: Path) -> Path:
    return root if root.is_absolute() else REPO_ROOT / root


KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{locator}"

CHUNK = 1 << 20


def kaggle_credentials() -> tuple[str, str]:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not path.is_file():
        raise RuntimeError(
            f"no Kaggle credentials at {path}; create a token at kaggle.com/settings, "
            "then save the downloaded kaggle.json there"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return payload["username"], payload["key"]
    except KeyError as exc:
        raise RuntimeError(f"{path} is missing the {exc} field") from exc


STALE_LOCK_SECONDS = 120


class DownloadLock:
    def __init__(self, target: Path):
        self.path = target.with_suffix(target.suffix + ".lock")
        self.handle: int | None = None

    def __enter__(self) -> DownloadLock:
        try:
            self.handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - self.path.stat().st_mtime
            if age < STALE_LOCK_SECONDS:
                owner = self.path.read_text(encoding="utf-8").strip() or "unknown"
                raise RuntimeError(
                    f"process {owner} is downloading {self.path.stem} right now "
                    f"(lock touched {age:.0f}s ago); let it finish rather than starting a second one"
                ) from None
            print(
                f"  clearing a stale lock, last touched {age / 60:.0f} min ago",
                file=sys.stderr,
            )
            self.path.unlink(missing_ok=True)
            self.handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self.handle, str(os.getpid()).encode())
        return self

    def touch(self) -> None:
        os.utime(self.path, None)

    def __exit__(self, *exc_info: object) -> None:
        if self.handle is not None:
            os.close(self.handle)
        self.path.unlink(missing_ok=True)


def stream_to_file(url: str, auth: tuple[str, str], target: Path, attempts: int = 20) -> Path:
    import requests

    partial = target.with_suffix(target.suffix + ".part")
    with DownloadLock(target) as lock:
        for attempt in range(1, attempts + 1):
            try:
                _stream_once(url, auth, partial, lock.touch)
                break
            except (requests.RequestException, OSError) as exc:
                if attempt == attempts:
                    raise
                have = partial.stat().st_size if partial.is_file() else 0
                print(
                    f"  attempt {attempt} stopped at {have / 1e9:.2f} GB ({exc}); resuming",
                    file=sys.stderr,
                )
                time.sleep(min(30, 2**attempt))
        partial.replace(target)
    return target


def _stream_once(
    url: str, auth: tuple[str, str], target: Path, heartbeat: Callable[[], None] | None = None
) -> Path:
    import requests

    resume = target.stat().st_size if target.is_file() else 0
    headers = {"Range": f"bytes={resume}-"} if resume else {}

    with requests.Session() as session:
        session.auth = auth
        response = session.get(url, headers=headers, stream=True, allow_redirects=False, timeout=60)
        hops = 0
        while response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            location = response.headers["Location"]
            response.close()
            hops += 1
            if hops > 10:
                raise RuntimeError(f"too many redirects fetching {url}")
            response = session.get(
                location, headers=headers, stream=True, allow_redirects=False, timeout=60
            )

        if resume and response.status_code == 200:
            resume = 0
        response.raise_for_status()

        expected = int(response.headers.get("Content-Length", 0)) + resume
        if resume and expected and resume > expected:
            response.close()
            target.unlink()
            raise OSError(
                f"local file is {resume} bytes but the source is {expected}; discarding and restarting"
            )

        digest = hashlib.md5() if not resume else None
        mode = "ab" if resume else "wb"
        done = resume
        last = -1.0
        with response, target.open(mode) as handle:
            for chunk in response.iter_content(CHUNK):
                handle.write(chunk)
                if digest is not None:
                    digest.update(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last > 2.0:
                    last = now
                    if heartbeat is not None:
                        heartbeat()
                    if expected:
                        print(
                            f"  {done / 1e9:.2f}/{expected / 1e9:.2f} GB "
                            f"({100 * done / expected:.1f}%)",
                            file=sys.stderr,
                        )
                    else:
                        print(f"  {done / 1e9:.2f} GB", file=sys.stderr)

        size = target.stat().st_size
        if expected and size != expected:
            raise OSError(f"got {size} bytes, expected {expected}")
        verify_md5(response.headers.get("x-goog-hash", ""), digest, target)
    return target


def verify_md5(goog_hash: str, digest: object, target: Path) -> None:
    if digest is None:
        return
    wanted = ""
    for part in goog_hash.split(","):
        name, _, value = part.strip().partition("=")
        if name == "md5":
            wanted = value + "=" * (-len(value) % 4)
    if not wanted:
        return
    got = base64.b64encode(digest.digest()).decode()
    if got != wanted:
        target.unlink(missing_ok=True)
        raise OSError(f"md5 mismatch for {target.name}: got {got}, expected {wanted}")


def fetch_kaggle(locator: str, dest: Path) -> None:
    auth = kaggle_credentials()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / (locator.split("/")[-1] + ".zip")
    stream_to_file(KAGGLE_DOWNLOAD_URL.format(locator=locator), auth, archive)
    extract_archives(dest)


def fetch_hf(locator: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=locator,
        repo_type="dataset",
        local_dir=str(dest),
        max_workers=4,
    )
    extract_archives(dest)


def missing_members(zf: zipfile.ZipFile, target: Path) -> list[zipfile.ZipInfo]:
    pending = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        local = target / info.filename
        if not local.is_file() or local.stat().st_size != info.file_size:
            pending.append(info)
    return pending


def extract_archives(dest: Path) -> list[Path]:
    extracted = []
    for archive in sorted(dest.rglob("*.zip")):
        target = archive.with_suffix("")
        try:
            with zipfile.ZipFile(archive) as zf:
                pending = missing_members(zf, target)
                if not pending:
                    continue
                broken = zf.testzip()
                if broken is not None:
                    raise zipfile.BadZipFile(f"corrupt entry {broken}")
                total = sum(1 for i in zf.infolist() if not i.is_dir())
                print(
                    f"extracting {archive.name}: {len(pending)} of {total} members missing",
                    file=sys.stderr,
                )
                target.mkdir(parents=True, exist_ok=True)
                zf.extractall(target, members=pending)
                remaining = missing_members(zf, target)
                if remaining:
                    raise RuntimeError(
                        f"{archive.name}: {len(remaining)} members still missing after extraction"
                    )
        except zipfile.BadZipFile as exc:
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                f"{archive.name} is corrupt ({exc}) and has been deleted; run the fetch again"
            ) from exc
        extracted.append(target)
        print(f"extracted {archive.name} -> {target.relative_to(dest)}")
    return extracted


def fetch(config: DatasetConfig, force: bool = False) -> str:
    dest = _resolve(config.root)
    kind, _, locator = config.source.partition(":")

    if kind == "manual":
        instructions = MANUAL_INSTRUCTIONS[config.name].format(url=config.source_url, dest=dest)
        return "manual\n" + instructions

    if count_audio(dest) and not force:
        extract_archives(dest)
        return f"ok ({count_audio(dest)} audio files under {config.root}, archive already fetched)"

    if kind == "kaggle":
        fetch_kaggle(locator, dest)
    elif kind == "hf":
        fetch_hf(locator, dest)
    else:
        raise RuntimeError(f"{config.name}: unknown source kind {kind!r}")

    return f"ok ({count_audio(dest)} audio files under {config.root})"


def main(argv: list[str] | None = None) -> int:
    datasets = available_datasets()
    parser = argparse.ArgumentParser(description="Fetch a PHASE dataset into data/raw")
    parser.add_argument("--dataset", choices=sorted(datasets), action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    selected = sorted(datasets) if args.all or not args.dataset else args.dataset

    if args.check:
        for key in selected:
            config = datasets[key]
            print(f"{key:20s} {count_audio(_resolve(config.root)):>7d} files  {config.root}")
        return 0

    failures = 0
    for key in selected:
        config = datasets[key]
        print(f"==> {key} ({config.source})")
        try:
            print(fetch(config, force=args.force))
        except Exception as exc:
            failures += 1
            print(f"FAILED: {exc}", file=sys.stderr)
        print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
