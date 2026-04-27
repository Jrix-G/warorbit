"""Scraping Kaggle : récupère le code/replays des top players d'Orbit Wars.

Usage:
    python -m analysis.scrape --notebooks
        Liste les top notebooks publics (besoin auth Kaggle).
    python -m analysis.scrape --download structured-baseline
        Download d'un notebook spécifique.
    python -m analysis.scrape --kovi-loss
        Cherche la seule défaite de kovi (#1 leaderboard).

Prérequis:
    pip install kaggle==2.0.2
    Token Kaggle dans %USERPROFILE%/.kaggle/kaggle.json
    (créer via kaggle.com/<user>/account → Create New API Token)
"""

import argparse
import json
import sys
from pathlib import Path


COMPETITION = "orbit-wars"
TARGETS = {
    "structured-baseline": "pilkwangkim/structured-baseline",   # à confirmer
    "orbit-star-wars": "TODO/orbit-star-wars-lb-max-1224",
    "distance-priority": "TODO/distance-prioritized",
    "sun-dodging": "TODO/sun-dodging-baseline",
}


def _have_kaggle():
    try:
        import kaggle  # noqa: F401
        return True
    except Exception:
        return False


def list_notebooks():
    if not _have_kaggle():
        print("Installe d'abord: pip install kaggle==2.0.2")
        return
    from kaggle import api
    api.authenticate()
    notebooks = api.kernels_list(competition=COMPETITION, sort_by="voteCount",
                                  page_size=20)
    print(f"Top notebooks pour {COMPETITION}:")
    for nb in notebooks:
        print(f"  votes={getattr(nb, 'totalVotes', '?'):>4}  "
              f"{nb.ref:50s}  {nb.title}")


def download_notebook(slug, out_dir="analysis/scraped"):
    if not _have_kaggle():
        print("Installe d'abord: pip install kaggle==2.0.2")
        return
    from kaggle import api
    api.authenticate()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    api.kernels_pull(slug, path=out_dir, metadata=True)
    print(f"Notebook {slug} téléchargé dans {out_dir}/")


def find_kovi_loss():
    """Liste les replays publics de kovi, cherche les défaites."""
    if not _have_kaggle():
        print("Installe d'abord: pip install kaggle==2.0.2")
        return
    from kaggle import api
    api.authenticate()
    print("Récupération des replays récents (TODO: vérifier API exacte)…")
    # Endpoint réel à vérifier: api.competition_submissions(...) ou
    # l'endpoint Episode/Match list est interne.
    # → workaround: scraper la page web /competitions/orbit-wars/leaderboard
    # avec le nom 'kovi' pour trouver match IDs.
    print("TODO: implémenter via web scraping si API absente.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--notebooks", action="store_true", help="Liste top notebooks")
    p.add_argument("--download", type=str, default=None, help="Slug notebook à dl")
    p.add_argument("--kovi-loss", action="store_true")
    args = p.parse_args()

    if args.notebooks:
        list_notebooks()
    elif args.download:
        slug = TARGETS.get(args.download, args.download)
        download_notebook(slug)
    elif args.kovi_loss:
        find_kovi_loss()
    else:
        p.print_help()
