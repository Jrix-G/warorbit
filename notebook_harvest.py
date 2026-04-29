"""Shared utilities for harvesting Orbit Wars notebook agents.

The repo already stores extracted notebook agents under `opponents/notebook_*.py`.
This module centralizes discovery, download, extraction, and module-name
sanitization so the scripts stop being hardcoded around the first four notebooks.
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from requests.cookies import create_cookie

DEFAULT_COMPETITION = "orbit-wars"
DEFAULT_BROWSER_AUTH = Path(__file__).resolve().parent / "replay_observer/output/kaggle_auth.json"
DEFAULT_KAGGLE_JSON = Path(__file__).resolve().parent / "kaggle.json"


def safe_module_name(ref: str) -> str:
    ref = ref.strip().replace("/", "_")
    ref = re.sub(r"[^a-zA-Z0-9_]+", "_", ref)
    ref = re.sub(r"_+", "_", ref).strip("_")
    if not ref.startswith("notebook_"):
        ref = f"notebook_{ref}"
    return ref.lower()


def _resolve_auth_path(auth_file: Optional[Path] = None) -> Path:
    if auth_file is not None:
        return auth_file
    env_path = os.environ.get("KAGGLE_BROWSER_AUTH")
    if env_path:
        return Path(env_path)
    return DEFAULT_BROWSER_AUTH


def _load_browser_auth(auth_file: Optional[Path] = None) -> Dict[str, Any]:
    path = _resolve_auth_path(auth_file)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _browser_api_token(auth_file: Optional[Path] = None) -> Optional[str]:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return os.environ["KAGGLE_API_TOKEN"]

    credentials_file = Path.home() / ".kaggle" / "access_token"
    if credentials_file.exists():
        token = credentials_file.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def _bootstrap_legacy_kaggle_auth() -> bool:
    """Populate KAGGLE_USERNAME/KAGGLE_KEY from a local kaggle.json if present."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    candidates = [
        Path(os.environ["KAGGLE_CONFIG_FILE"]).expanduser()
        if os.environ.get("KAGGLE_CONFIG_FILE")
        else None,
        DEFAULT_KAGGLE_JSON,
        Path.home() / ".kaggle" / "kaggle.json",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        username = data.get("username")
        key = data.get("key")
        if username and key:
            os.environ.setdefault("KAGGLE_USERNAME", username)
            os.environ.setdefault("KAGGLE_KEY", key)
            return True
    return False


def _browser_session(auth_file: Optional[Path] = None) -> Optional[requests.Session]:
    auth = _load_browser_auth(auth_file)
    token = _browser_api_token(auth_file)
    if not auth and not token:
        return None

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "kaggle-api/v1.7.0",
            "Content-Type": "application/json",
        }
    )

    cookies = auth.get("cookies", [])
    xsrf_token = None
    for cookie in cookies:
        name = cookie.get("name")
        value = cookie.get("value")
        if not name or value is None:
            continue
        domain = cookie.get("domain") or "www.kaggle.com"
        path = cookie.get("path") or "/"
        try:
            session.cookies.set_cookie(
                create_cookie(
                    name=name,
                    value=value,
                    domain=domain,
                    path=path,
                )
            )
        except Exception:
            continue
        if name in {"XSRF-TOKEN", "CSRF-TOKEN"} and value:
            xsrf_token = value

    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    if xsrf_token:
        session.headers["X-XSRF-TOKEN"] = xsrf_token

    return session


def _post_kaggle_json(endpoint: str, payload: Dict[str, Any], auth_file: Optional[Path] = None) -> Dict[str, Any]:
    session = _browser_session(auth_file)
    if session is None:
        raise RuntimeError("No Kaggle auth available")
    response = session.post(endpoint, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("code", 200) >= 400:
        raise RuntimeError(data.get("message") or f"Kaggle request failed: {endpoint}")
    return data


def _kernel_api_endpoint(method: str) -> str:
    return f"https://www.kaggle.com/api/i/kernels.KernelsApiService/{method}"


def _kernel_list_request_dict(
    competition: str,
    page: int = 1,
    page_size: int = 100,
    page_token: str = "",
) -> Dict[str, Any]:
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelsRequest
    from kagglesdk.kernels.types.kernels_enums import KernelsListSortType, KernelsListViewType

    request = ApiListKernelsRequest()
    request.competition = competition
    request.group = KernelsListViewType.EVERYONE
    request.language = "all"
    request.kernel_type = "all"
    request.output_type = "all"
    request.sort_by = KernelsListSortType.VOTE_COUNT
    request.page = page
    request.page_size = page_size
    request.page_token = page_token
    return request.to_dict()


def _kernel_pull_request_dict(kernel_ref: str) -> Dict[str, Any]:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    user_name, kernel_slug = kernel_ref.split("/", 1)
    request = ApiGetKernelRequest()
    request.user_name = user_name
    request.kernel_slug = kernel_slug
    return request.to_dict()


def _kernel_to_dict(kernel: Any) -> Dict[str, Any]:
    return {
        "ref": getattr(kernel, "ref", None),
        "title": getattr(kernel, "title", None),
        "totalVotes": getattr(kernel, "total_votes", getattr(kernel, "totalVotes", None)),
        "author": getattr(kernel, "author", None),
        "slug": getattr(kernel, "slug", None),
        "id": getattr(kernel, "id", None),
    }


def _strip_notebook_magics(source: str) -> str:
    lines = source.splitlines()
    cleaned: List[str] = []
    skip_first = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if i == 0 and stripped.startswith("%%"):
            # Keep the cell body, drop the cell magic.
            skip_first = True
            continue
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        cleaned.append(line)
    if skip_first and not cleaned:
        return ""
    return "\n".join(cleaned).strip("\n")


def notebook_code(nb: Dict[str, Any]) -> str:
    code_parts: List[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        cleaned = _strip_notebook_magics(source)
        if cleaned.strip():
            code_parts.append(cleaned)
    return "\n\n".join(code_parts)


_AGENT_RE = re.compile(r"^\s*def\s+(agent|make_agent|my_agent)\s*\(", re.M)
_WRITEFILE_RE = re.compile(r"^\s*%%writefile\s+(\S+)\s*$")


def extract_agent_module_source(nb_path: Path) -> Tuple[Optional[str], List[str]]:
    """Return (module_source, imports) for the first notebook agent definition.

    We keep all code cells from the top of the notebook up to and including the
    cell that defines the agent entry point. This preserves helper functions
    defined earlier in the notebook while avoiding later demo/training cells.
    """
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    code_parts: List[str] = []
    imports: List[str] = []
    agent_found = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        raw_first_line = source.splitlines()[0] if source.splitlines() else ""
        cleaned = _strip_notebook_magics(source)
        if not cleaned.strip():
            continue
        if re.search(r"^\s*(import|from)\s+", cleaned, re.M):
            imports.append(cleaned)
        code_parts.append(cleaned)
        if _WRITEFILE_RE.match(raw_first_line):
            agent_found = True
            break
        if _AGENT_RE.search(cleaned):
            agent_found = True
            break

    if not agent_found:
        return None, imports

    return "\n\n".join(code_parts), imports


def extract_agent_code(nb_path: Path) -> Tuple[Optional[str], List[str]]:
    module_source, imports = extract_agent_module_source(nb_path)
    if module_source is None:
        return None, imports
    return module_source, imports


def discover_competition_kernels(
    competition: str = DEFAULT_COMPETITION,
    limit: int = 50,
    auth_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return public notebook kernels for a competition, sorted by votes.

    Uses the browser auth token from `replay_observer/output/kaggle_auth.json`
    when present, and falls back to Kaggle's Python API only if needed.
    """
    if _bootstrap_legacy_kaggle_auth():
        try:
            from kaggle import api

            api.authenticate()
            kernels = api.kernels_list(competition=competition, sort_by="voteCount", page_size=limit)
            return [_kernel_to_dict(k) for k in kernels]
        except Exception as exc:
            print(f"Legacy Kaggle API discovery failed: {exc}")

    session = _browser_session(auth_file)
    if session is not None:
        try:
            payload = _kernel_list_request_dict(competition, page=1, page_size=max(100, limit))
            data = _post_kaggle_json(_kernel_api_endpoint("ListKernels"), payload, auth_file=auth_file)
            from kagglesdk.kernels.types.kernels_api_service import ApiListKernelsResponse

            response = ApiListKernelsResponse.from_dict(data)
            kernels = [_kernel_to_dict(kernel) for kernel in (response.kernels or [])]
            if kernels:
                return kernels[:limit]
        except Exception as exc:
            print(f"Browser kernel discovery failed: {exc}")

    return []


def _download_dir_candidates(base_dir: Path, kernel_ref: str) -> List[Path]:
    slug = kernel_ref.split("/")[-1]
    author = kernel_ref.split("/")[0]
    return [
        base_dir / f"{slug}.ipynb",
        base_dir / f"{author}_{slug}.ipynb",
        base_dir / slug,
    ]


def download_kernel_notebook(
    kernel_ref: str,
    output_dir: Path,
    refresh: bool = False,
    auth_file: Optional[Path] = None,
) -> Optional[Path]:
    """Download a Kaggle notebook kernel to `output_dir` and return the file path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    slug = kernel_ref.split("/")[-1]
    existing = [p for p in _download_dir_candidates(output_dir, kernel_ref) if p.exists()]
    if existing and not refresh:
        return existing[0]

    if _bootstrap_legacy_kaggle_auth():
        try:
            from kaggle import api

            before = {p.resolve() for p in output_dir.glob("**/*.ipynb")}
            api.kernels_pull(kernel_ref, path=str(output_dir), metadata=True)
            after = [p for p in output_dir.glob("**/*.ipynb") if p.resolve() not in before]
            if after:
                return max(after, key=lambda p: p.stat().st_mtime)

            for candidate in _download_dir_candidates(output_dir, kernel_ref):
                if candidate.exists():
                    return candidate
            all_downloaded = sorted(output_dir.glob("**/*.ipynb"), key=lambda p: p.stat().st_mtime, reverse=True)
            if all_downloaded:
                return all_downloaded[0]
        except Exception as exc:
            print(f"Legacy Kaggle API download failed for {kernel_ref}: {exc}")

    return None


def write_opponent_module(module_name: str, module_source: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{module_name}.py"
    header = [
        '"""Auto-extracted from a Kaggle notebook."""',
        "",
        module_source.strip(),
        "",
    ]
    path.write_text("\n".join(header), encoding="utf-8")
    return path
