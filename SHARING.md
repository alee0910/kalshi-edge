# Sharing the kalshi-edge report

This repo ships a zero-JS static HTML report of the current Kalshi market
universe we're tracking. Two ways to share it:

## 1. Send the HTML file directly

After running the CLI locally, you'll have a single self-contained file at
`site/index.html`. It has inline CSS, no external assets, no JS. Open it
in any browser, email it, drop it in Slack, put it on a USB stick.

```bash
uv run kalshi-edge init-db
uv run kalshi-edge pull-universe --series "KXHIGHNY,KXHIGHLAX,KXHIGHCHI,KXCPIYOY,KXCPI,KXFED,KXFEDDECISION,KXNBAGAME,KXNHL,KXATPMATCH,KXMLBGAME"
uv run kalshi-edge build-report --out site/index.html
open site/index.html
```

## 2. Host it on GitHub Pages with auto-refresh (recommended)

The workflow at `.github/workflows/update-report.yml` runs every 30 minutes:

1. Sync deps with `uv`
2. Pull the universe from Kalshi's public API
3. Rebuild the HTML
4. Deploy to GitHub Pages

**Public Kalshi endpoints don't require auth**, so no secrets are needed.

### One-time setup

1. Create a GitHub repo and push this codebase:
   ```bash
   cd /Users/Andrew/kalshi-edge
   git init && git add -A && git commit -m "initial scaffold"
   gh repo create kalshi-edge --public --source=. --remote=origin --push
   ```

2. Enable GitHub Pages for the repo:
   - Repo → Settings → Pages → **Source**: set to "GitHub Actions"
   - (This is the new GitHub-Actions-based Pages publishing flow; we don't
     need a `gh-pages` branch.)

3. The workflow triggers automatically on push to `main` and on the cron.
   You can also trigger a rebuild manually: Actions tab → "update-report"
   → "Run workflow".

Your URL will be `https://<username>.github.io/kalshi-edge/`. Share that.

### Cron schedule

Currently `*/30 * * * *` (every 30 min). To change, edit the `cron` line
in [.github/workflows/update-report.yml](.github/workflows/update-report.yml).
Note GitHub's scheduled workflows are **best-effort** and may drift by
several minutes under load.

### Series list

The workflow pulls a targeted set of Kalshi series (weather cities,
economic releases, sports). Edit the `--series` flag in the workflow to
widen or narrow coverage. A full-universe scan is possible
(`pull-universe` with no `--series` flag) but wastes request budget on
the thousands of multi-venue parlay markets that Kalshi returns first.

## What the report shows today

- Current universe: every contract passing the filter in `config/default.yaml`
  (min DTE, min 24h volume, max spread).
- Latest snapshot: YES bid/ask/mid/spread, 24h volume, liquidity.
- Grouped by category; top 50 per category shown by descending volume.

## What the report will show once forecasters land

Same page, same URL, but each category section gains columns for:

- `model p_yes` — our Bayesian posterior for YES
- `market p_yes` — the market mid
- `edge` — posterior minus market, in points
- `uncertainty` — 5%/95% posterior band, and epistemic-vs-aleatoric split
- `model_confidence` — a 0-1 scalar self-assessment

Forecaster rollout order: weather → economics → rates → sports → politics.
