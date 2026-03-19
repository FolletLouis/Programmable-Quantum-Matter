# Deploy to Hugging Face Spaces (Public Hosting)

Get a **free public URL** like `https://huggingface.co/spaces/YOUR_USERNAME/pqm-explorer`

---

## Step 1: Create a Hugging Face account

Sign up at [huggingface.co](https://huggingface.co/join) if you don't have one.

---

## Step 2: Create a new Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Fill in:
   - **Space name:** `pqm-explorer` (or any name)
   - **License:** MIT (or your choice)
   - **SDK:** Select **Docker** (not Gradio, Streamlit, or others)
   - **Visibility:** Public

4. Click **Create Space**

**Template:** There is no special "template" — just choose **Docker** as the SDK. You'll get a minimal Space; add your own files.

---

## Step 3: Push your files to the Space

Hugging Face Spaces use Git. You can push via:

### Option A: Git (recommended)

```bash
# Clone your new Space (replace YOUR_USERNAME with your HF username)
git clone https://huggingface.co/spaces/YOUR_USERNAME/pqm-explorer
cd pqm-explorer

# Copy these files from your Programmable_Quantum_Matter repo:
# - Dockerfile
# - requirements.txt
# - paper_explorer.py
# - custom.css
# - app/ (entire folder)
# - PQM_latex_file/figs/ (entire folder)
# - Temperature and T2 simulations/ (entire folder)
# - Entanglement simulations/ (entire folder)
# - README_HF_SPACE.md → rename to README.md (has app_port: 8080 for HF)

# Commit and push
git add .
git commit -m "Add Paper Explorer app"
git push
```

### Option B: Upload in the browser

1. Open your Space on the Hugging Face website
2. Click **Files** → **Add file** → **Upload files**
3. Upload: `Dockerfile`, `requirements.txt`, `paper_explorer.py`, `custom.css`
4. Upload the `app/` folder (zip it first, or upload files one by one)
5. Upload `PQM_latex_file/figs/` contents
6. Upload the CSV folders

---

## Step 4: Wait for the build

Hugging Face will detect the Dockerfile and build your app. This takes 5–10 minutes. You'll see a build log; when it finishes, your app will be live.

---

## Step 5: Share the URL

Your app will be at:
```
https://huggingface.co/spaces/YOUR_USERNAME/pqm-explorer
```

Share this link with collaborators. No login required to view.

---

## Troubleshooting

- **Build fails:** Check the build logs in your Space. Common issues: missing files, wrong paths in Dockerfile.
- **App loads but figures missing:** Ensure `PQM_latex_file/figs/` was uploaded with the PDF files.
- **"Run Filter_Function_II" message:** The CSV data folders may be empty. Upload the `N=2_results_parallel.csv` etc. from `Temperature and T2 simulations/` and `Entanglement simulations/`.

---

## Files checklist

| File/Folder | Required |
|-------------|----------|
| Dockerfile | Yes |
| requirements.txt | Yes |
| paper_explorer.py | Yes |
| custom.css | Yes |
| app/ | Yes |
| PQM_latex_file/figs/ | Yes |
| Temperature and T2 simulations/ | For T2 interactive plots |
| Entanglement simulations/ | For n_links, ε_eff plots |
