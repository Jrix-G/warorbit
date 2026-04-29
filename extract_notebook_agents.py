#!/usr/bin/env python3
"""Extract notebook agents into `opponents/notebook_*.py`.

By default this script scans local notebook files under `notebooks/`. With
`--discover`, it first asks Kaggle for the top competition kernels, downloads
them, and keeps extracting until it has `--limit` notebook agents or runs out
of usable kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from notebook_harvest import (
    DEFAULT_COMPETITION,
    DEFAULT_BROWSER_AUTH,
    discover_competition_kernels,
    download_kernel_notebook,
    extract_agent_code,
    safe_module_name,
    write_opponent_module,
)


if os.name == "nt":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


LOCAL_NOTEBOOKS = [
    "notebooks/orbitbotnext.ipynb",
    "notebooks/distance-prioritized-agent-lb-max-score-1100.ipynb",
    "notebooks/lb-928-7-physics-accurate-planner.ipynb",
    "notebooks/orbit-wars-2026-tactical-heuristic.ipynb",
]


def _local_ref_from_path(path: Path) -> str:
    stem = path.stem
    legacy = {
        "orbitbotnext": "notebook_orbitbotnext",
        "distance-prioritized-agent-lb-max-score-1100": "notebook_distance_prioritized",
        "lb-928-7-physics-accurate-planner": "notebook_physics_accurate",
        "orbit-wars-2026-tactical-heuristic": "notebook_tactical_heuristic",
    }
    return legacy.get(stem, stem)


def _iter_local_notebooks():
    for nb in LOCAL_NOTEBOOKS:
        path = Path(nb)
        if path.exists():
            yield path


def _extract_one(nb_path: Path, module_name: str, out_dir: Path) -> bool:
    module_source, _imports = extract_agent_code(nb_path)
    if not module_source:
        print(f"  No agent found")
        return False
    out_path = write_opponent_module(module_name, module_source, out_dir)
    lines = len(module_source.splitlines())
    print(f"  Extracted: {lines} lines")
    print(f"  Saved: {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract notebook agents from Kaggle notebooks.")
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--discover", action="store_true",
                        help="discover kernels from Kaggle and download them before extraction")
    parser.add_argument("--download-dir", default="notebooks/harvested")
    parser.add_argument("--output-dir", default="opponents")
    parser.add_argument("--manifest", default="opponents/notebook_manifest.json")
    parser.add_argument("--auth-file", default=str(DEFAULT_BROWSER_AUTH),
                        help="browser auth export containing Kaggle cookies and client token")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    download_dir = Path(args.download_dir)
    manifest_path = Path(args.manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting notebook agents...")
    print("=" * 80)

    manifest = []
    extracted = 0

    if args.discover:
        kernels = discover_competition_kernels(
            args.competition,
            limit=max(50, args.limit * 3),
            auth_file=Path(args.auth_file),
        )
        if not kernels:
            raise SystemExit(
                "Could not discover Kaggle kernels. Configure KAGGLE_API_TOKEN or the browser auth export."
            )
        print(f"Discovered {len(kernels)} kernels for {args.competition}")
        for kernel in kernels:
            if extracted >= args.limit:
                break
            ref = kernel.get("ref")
            if not ref:
                continue
            module_name = safe_module_name(ref)
            nb_path = download_kernel_notebook(
                ref,
                download_dir,
                refresh=args.refresh,
                auth_file=Path(args.auth_file),
            )
            if nb_path is None or not nb_path.exists():
                continue
            print(f"\n{module_name} ({ref}):")
            ok = _extract_one(nb_path, module_name, out_dir)
            manifest.append(
                {
                    "ref": ref,
                    "module": module_name,
                    "notebook": str(nb_path),
                    "ok": ok,
                    "votes": kernel.get("totalVotes"),
                    "title": kernel.get("title"),
                }
            )
            if ok:
                extracted += 1
    else:
        for nb_path in _iter_local_notebooks():
            module_name = safe_module_name(_local_ref_from_path(nb_path))
            print(f"\n{module_name}:")
            ok = _extract_one(nb_path, module_name, out_dir)
            manifest.append(
                {
                    "ref": module_name,
                    "module": module_name,
                    "notebook": str(nb_path),
                    "ok": ok,
                    "title": nb_path.name,
                }
            )
            if ok:
                extracted += 1

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"Extracted {extracted} notebook agents")
    print(f"Manifest: {manifest_path}")
    print("Next: import the new notebook_* modules through opponents/__init__.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
