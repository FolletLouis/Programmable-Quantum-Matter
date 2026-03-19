# Deploying the Paper Explorer

Share the interactive Paper Explorer with collaborators via Hugging Face Spaces (easiest), Railway, or Docker.

---

## Quick start (local test)

```bash
docker build -t pqm-explorer .
docker run -p 8080:8080 pqm-explorer
```

Open http://localhost:8080

---

## Option 1: Hugging Face Spaces (recommended)

**Free, public URL, no server to manage.**

1. **Create a Space** at [huggingface.co/spaces](https://huggingface.co/spaces)
   - Choose **Docker** as the SDK (not Gradio)
   - Name it e.g. `pqm-paper-explorer`

2. **Clone your Space** and push these files:

   ```
   your-space/
   ├── Dockerfile          # Use the one in this repo
   ├── requirements.txt    # Use the one in this repo
   ├── paper_explorer.py
   ├── custom.css
   ├── app/
   ├── PQM_latex_file/figs/
   ├── Temperature and T2 simulations/
   └── Entanglement simulations/
   ```

3. **Or** fork [marimo-team/marimo-app-template](https://huggingface.co/spaces/marimo-team/marimo-app-template), then replace `app.py` with `paper_explorer.py` and add the `app/`, `PQM_latex_file/figs/`, and data folders.

4. HF will build and serve at `https://huggingface.co/spaces/YOUR_USERNAME/pqm-paper-explorer`

---

## Option 2: Railway

1. Install [Railway CLI](https://docs.railway.app/develop/cli) or use the web dashboard.

2. From the repo root:
   ```bash
   railway init
   railway up
   ```

3. Railway will detect the Dockerfile and deploy. Set the public URL in the dashboard.

---

## Option 3: Docker (self-host or any cloud)

**Build and run locally:**

```bash
docker build -t pqm-explorer .
docker run -p 8080:8080 pqm-explorer
```

Open http://localhost:8080

**Deploy to a VPS (AWS, GCP, Azure, etc.):**

1. Push the image to a registry (Docker Hub, GHCR, etc.):
   ```bash
   docker tag pqm-explorer your-registry/pqm-explorer:latest
   docker push your-registry/pqm-explorer:latest
   ```

2. On your server:
   ```bash
   docker pull your-registry/pqm-explorer:latest
   docker run -d -p 8080:8080 --name pqm your-registry/pqm-explorer:latest
   ```

3. Put a reverse proxy (nginx, Caddy) in front for HTTPS.

---

## Files needed for deployment

| Path | Purpose |
|------|---------|
| `paper_explorer.py` | Main app |
| `custom.css` | Styling |
| `app/` | Python package (figures, data loader, etc.) |
| `PQM_latex_file/figs/` | PDF figures |
| `Temperature and T2 simulations/` | CSV data for T2 plots |
| `Entanglement simulations/` | CSV data for n_links, ε_eff |
| `requirements.txt` | Dependencies |

The app degrades gracefully if CSV folders are missing (some interactive figures will show a message to run the simulation scripts).

---

## Minimal deploy (figures only)

If you want a smaller image and can skip T2/entanglement interactivity:

1. Remove the two `COPY` lines for `Temperature and T2 simulations` and `Entanglement simulations` from the Dockerfile.
2. Add `RUN mkdir -p "Temperature and T2 simulations" "Entanglement simulations"` so the app doesn't error on missing paths.

---

## Sharing with collaborators

- **Hugging Face**: Share the Space URL. Anyone can view; no login required (unless you restrict it).
- **Railway**: Share the deployed URL.
- **Docker**: Share the image or run `docker compose` for a one-command local setup.
