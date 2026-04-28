#!/usr/bin/env python3
"""Automated Kaggle replay observer.

The script first tries to capture real data from the browser/network. If Kaggle
does not expose structured replay data, it still records screenshots for later
vision/OCR work.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output"
SUSPECT_RE = re.compile(
    r"(episode|replay|orbit|observation|planets|fleets|leaderboard|submission)",
    re.IGNORECASE,
)


def safe_name(value: str, limit: int = 120) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return value[:limit] or "item"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def preview_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... truncated {len(text) - limit} chars ..."


def observe(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is not installed. Run:\n"
            "  python3 -m pip install -r replay_observer/requirements.txt\n"
            "  python3 -m playwright install chromium"
        ) from exc

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / run_id
    screenshots_dir = out_dir / "screenshots"
    canvas_dir = out_dir / "canvas"
    network_dir = out_dir / "network"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    canvas_dir.mkdir(parents=True, exist_ok=True)
    network_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "run_id": run_id,
        "url": args.url,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "seconds": args.seconds,
        "interval": args.interval,
        "network_hits": [],
        "json_responses": [],
        "screenshots": [],
        "canvas_screenshots": [],
        "console": [],
        "errors": [],
    }

    with sync_playwright() as p:
        context_kwargs = {
            "viewport": {"width": args.width, "height": args.height},
            "device_scale_factor": 1,
        }
        if args.storage:
            context_kwargs["storage_state"] = args.storage

        launch_args = [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-blink-features=AutomationControlled",
        ]
        launch_kwargs = {
            "headless": not args.headed,
            "slow_mo": args.slow_mo,
            "args": launch_args,
        }
        if args.channel:
            launch_kwargs["channel"] = args.channel

        if args.user_data_dir:
            context = p.chromium.launch_persistent_context(
                args.user_data_dir,
                **launch_kwargs,
                **context_kwargs,
            )
            browser = context.browser
            page = context.pages[0] if context.pages else context.new_page()
        else:
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context(**context_kwargs)
            page = context.new_page()

        def on_console(msg):
            report["console"].append({
                "type": msg.type,
                "text": msg.text[:2000],
            })

        def on_page_error(exc):
            report["errors"].append({"pageerror": str(exc)})

        def on_response(response):
            url = response.url
            content_type = response.headers.get("content-type", "")
            is_suspect = bool(SUSPECT_RE.search(url)) or "json" in content_type.lower()
            if not is_suspect:
                return

            hit = {
                "url": url,
                "status": response.status,
                "content_type": content_type,
            }
            report["network_hits"].append(hit)

            if "json" not in content_type.lower() and not SUSPECT_RE.search(url):
                return

            try:
                body = response.text()
            except Exception as exc:
                hit["body_error"] = str(exc)
                return

            name = safe_name(f"{len(report['network_hits']):04d}_{url}")
            body_path = network_dir / f"{name}.txt"
            body_path.write_text(preview_text(body, args.max_body_chars), encoding="utf-8")
            hit["body_file"] = str(body_path)

            try:
                parsed = json.loads(body)
            except Exception:
                return

            json_path = network_dir / f"{name}.json"
            write_json(json_path, parsed)
            report["json_responses"].append({
                "url": url,
                "status": response.status,
                "file": str(json_path),
                "top_level_type": type(parsed).__name__,
                "top_level_keys": list(parsed.keys())[:80] if isinstance(parsed, dict) else None,
            })

        page.on("console", on_console)
        page.on("pageerror", on_page_error)
        page.on("response", on_response)

        print(f"[observer] opening {args.url}")
        page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except PlaywrightTimeoutError:
            report["errors"].append({"networkidle_timeout": True})

        if args.login_wait > 0:
            print(f"[observer] login/manual wait: {args.login_wait}s")
            time.sleep(args.login_wait)

        probe_path = ROOT / "browser_probe.js"
        probe_source = probe_path.read_text(encoding="utf-8")

        started = time.perf_counter()
        capture_idx = 0
        while time.perf_counter() - started <= args.seconds:
            stamp = f"{capture_idx:04d}"
            screenshot_path = screenshots_dir / f"page_{stamp}.png"
            page.screenshot(path=str(screenshot_path), full_page=args.full_page)
            report["screenshots"].append(str(screenshot_path))

            try:
                probe = page.evaluate(probe_source)
                report["page_probe"] = probe
            except Exception as exc:
                report["errors"].append({"probe_error": str(exc)})

            canvases = page.locator("canvas")
            try:
                count = canvases.count()
            except Exception:
                count = 0
            for i in range(min(count, args.max_canvases)):
                canvas_path = canvas_dir / f"canvas_{stamp}_{i}.png"
                try:
                    canvases.nth(i).screenshot(path=str(canvas_path))
                    report["canvas_screenshots"].append(str(canvas_path))
                except Exception as exc:
                    report["errors"].append({"canvas_error": str(exc), "index": i})

            print(
                f"[observer] capture={capture_idx} "
                f"network_hits={len(report['network_hits'])} "
                f"json={len(report['json_responses'])} "
                f"canvas={len(report['canvas_screenshots'])}"
            )
            capture_idx += 1
            time.sleep(args.interval)

        if args.save_storage:
            context.storage_state(path=args.save_storage)
            report["saved_storage"] = args.save_storage

        report["finished_at"] = datetime.utcnow().isoformat() + "Z"
        write_json(out_dir / "report.json", report)
        context.close()
        if browser is not None and not args.user_data_dir:
            browser.close()

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observe a Kaggle Orbit Wars replay.")
    parser.add_argument("--url", required=True, help="Kaggle replay URL.")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-id")
    parser.add_argument("--headed", action="store_true", help="Show browser window.")
    parser.add_argument("--login-wait", type=float, default=0.0, help="Manual wait after page load.")
    parser.add_argument("--storage", help="Playwright storage_state JSON to reuse login.")
    parser.add_argument("--save-storage", help="Path to save Playwright storage_state JSON.")
    parser.add_argument("--user-data-dir", help="Persistent browser profile directory.")
    parser.add_argument("--channel", help="Browser channel, e.g. chrome or msedge if installed.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--full-page", action="store_true")
    parser.add_argument("--max-canvases", type=int, default=4)
    parser.add_argument("--max-body-chars", type=int, default=500_000)
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--slow-mo", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = observe(args)
    print("\n[observer] done")
    print(f"[observer] report: {Path(args.output) / report['run_id'] / 'report.json'}")
    print(f"[observer] network hits: {len(report['network_hits'])}")
    print(f"[observer] json responses: {len(report['json_responses'])}")
    print(f"[observer] screenshots: {len(report['screenshots'])}")
    print(f"[observer] canvas screenshots: {len(report['canvas_screenshots'])}")


if __name__ == "__main__":
    main()
