"""
Capture Orbit Wars replay JSON from Kaggle via Playwright.
Usage: python3 replay_observer.py <submissionId> <episodeId>
       python3 replay_observer.py 52128366 75588964
"""
import asyncio, json, sys, os, time
from playwright.async_api import async_playwright

SUBMISSION_ID = sys.argv[1] if len(sys.argv) > 1 else "52128366"
EPISODE_ID    = sys.argv[2] if len(sys.argv) > 2 else "75588964"
URL = f"https://www.kaggle.com/competitions/orbit-wars/submissions?submissionId={SUBMISSION_ID}&episodeId={EPISODE_ID}"
OUT_DIR = os.path.join(os.path.dirname(__file__), "replays")
os.makedirs(OUT_DIR, exist_ok=True)

captured = {}

async def handle_response(response):
    url = response.url
    if "GetEpisodeReplay" in url or "GetEpisode" in url or "ListEpisodes" in url:
        try:
            body = await response.json()
            key = url.split("/")[-1].split("?")[0]
            captured[key] = body
            print(f"  CAPTURED: {key} ({len(str(body))} chars)")
        except Exception as e:
            print(f"  FAILED to parse {url}: {e}")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=100)
        ctx = await browser.new_context()
        page = await ctx.new_page()
        page.on("response", handle_response)

        print("Step 1: navigate to Kaggle login")
        await page.goto("https://www.kaggle.com/account/login")
        print(">>> LOGIN MANUALLY IN THE BROWSER. Press Enter here when done <<<")
        input()

        print(f"Step 2: loading replay {EPISODE_ID}...")
        await page.goto(URL)
        print("Waiting 15s for API calls to complete...")
        await asyncio.sleep(15)

        if not captured:
            print("Nothing captured yet — waiting 15s more...")
            await asyncio.sleep(15)

        if captured:
            out_path = os.path.join(OUT_DIR, f"episode_{EPISODE_ID}.json")
            with open(out_path, "w") as f:
                json.dump(captured, f, indent=2)
            print(f"\nSaved {len(captured)} endpoints to {out_path}")

            # Also extract just the replay steps if present
            for k, v in captured.items():
                if "replay" in k.lower() or "step" in str(v)[:200].lower():
                    replay_path = os.path.join(OUT_DIR, f"replay_{EPISODE_ID}_steps.json")
                    with open(replay_path, "w") as f:
                        json.dump(v, f, indent=2)
                    print(f"Replay steps saved to {replay_path}")
                    break
        else:
            print("No API calls captured. Try navigating to the replay page manually.")

        await browser.close()

asyncio.run(main())
