# Deploying FairPath

Checked in November 2026. Free tiers change often, so confirm the limits on each provider's pricing page before relying on them.

| Option | Cost | What you get | Trade-off |
|---|---|---|---|
| **A. Vercel (dashboard) + Render (API)** — recommended | free | Dashboard always on and instant; audit pages work even while the API sleeps | Two services; first prediction after 15 idle minutes waits ~1 minute |
| B. Single Render web service (Docker) | free | One URL, one deploy | Whole site sleeps after 15 idle minutes; 512 MB RAM, 0.1 CPU |
| C. Hugging Face Spaces (Docker SDK) | needs a paid plan | 2 vCPU / 16 GB, ML-community visibility | Creating Docker Spaces now requires a PRO / Team plan |
| D. Your own machine or VPS | – | `docker compose up --build` | You run it |

Measured locally: the API uses about 260 MB of RAM after serving predictions and batch requests, so it fits Render's 512 MB free instance.
With 0.1 CPU a single prediction with explanation and counterfactuals takes noticeably longer than on a laptop, which is fine for a demo.

## Before anything: push the repository

See [GITHUB_SETUP.md](GITHUB_SETUP.md).

## Option A — Vercel + Render (recommended)

**1. API on Render**

1. Sign in at <https://render.com> with GitHub. (Render may ask for a card even on the free plan.)
2. **New → Blueprint**, pick your `fairpath` repository. Render reads `render.yaml` and creates a free Docker web service with
   `/api/health` as its health check.
3. Wait for the first build (around 10 minutes), then open `https://<your-service>.onrender.com/api/health`. You should see `"status":"ok"`.
4. Optional but tidy: in the service's **Environment**, set `FAIRPATH_CORS_ORIGINS` to your Vercel URL once you have it.

**2. Dashboard on Vercel**

1. Sign in at <https://vercel.com> with GitHub, **Add New → Project**, import the repository.
2. Set **Root Directory** to `web`. Vercel detects Next.js; the static export is produced by `npm run build`.
3. Add environment variables:
   * `NEXT_PUBLIC_API_BASE` = `https://<your-service>.onrender.com`
   * `NEXT_PUBLIC_REPO_URL` = `https://github.com/SadrulCrafts/fairpath`
4. Deploy. Your demo URL is `https://<project>.vercel.app`. Put it in the README badges.

The audit pages (overview, fairness, models, methodology) read `web/public/data/*.json`, which is committed, so they never wait for the API.
Prediction pages show a "waking up the model server" message if Render was asleep.

*Keeping it warm (optional):* a free monitor such as UptimeRobot or cron-job.org pinging `/api/health` every 10 minutes avoids most cold
starts. Pinging uses free instance hours (750 per month on Render); one service pinged around the clock stays inside that.

## Option B — one Render service

Same as step A.1. The Docker image serves the dashboard at `/` and the API at `/api`, so `https://<your-service>.onrender.com` is the
whole demo. No Vercel needed.

## Option C — Hugging Face Spaces (paid plan)

1. Create a Space, SDK **Docker**, hardware CPU Basic.
2. Push this repository to the Space's git remote, with a `README.md` at the root that starts with this front matter (keep the rest of the
   README below it):
   ```yaml
   ---
   title: FairPath
   emoji: 🎓
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   pinned: false
   license: mit
   ---
   ```
3. The Space builds the Dockerfile and serves on port 7860.

## Updating after you retrain

```bash
make train && make report      # regenerates reports, figures, RESULTS.md and web/public/data
git add reports artifacts docs web/public/data && git commit -m "chore(results): retrain" && git push
```
Both Render and Vercel redeploy automatically on push.

## Production notes

* Put the API behind HTTPS (both hosts do this for you) and restrict `FAIRPATH_CORS_ORIGINS` to your dashboard's origin.
* The API is stateless and stores nothing, but your host's access logs will record request paths and IPs. Choose a region and log
  retention that fit your institution's privacy rules.
* Re-run the full audit on local data before any real use (see MODEL_CARD.md).
