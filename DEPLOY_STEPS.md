# Complete Steps: Deploy Marimo Paper Explorer to Hugging Face

This guide walks you through deploying your Marimo notebook (`paper_explorer.py`) to Hugging Face Spaces so it's publicly accessible.

---

## Prerequisites

- Hugging Face account ([huggingface.co/join](https://huggingface.co/join))
- Git installed
- Git LFS installed ([git-lfs.com](https://git-lfs.com))

---

## Step 1: Create a Hugging Face Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name:** `Programmable-Quantum-Matter` (or any name)
   - **License:** MIT
   - **SDK:** Click **Docker**
   - **Template:** Select **marimo** (or **Blank**)
   - **Visibility:** **Public**
3. Click **Create Space**
4. Note your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

---

## Step 2: Create a Hugging Face Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **Create new token**
3. Name it (e.g. `spaces-deploy`)
4. Role: **Write**
5. Click **Generate**
6. **Copy the token** and store it securely (you won't see it again)

---

## Step 3: Prepare the Deploy Folder

Open PowerShell in your project folder and run:

```powershell
cd c:\Users\anand43\Documents\Programmable_Quantum_Matter
.\prepare_hf_deploy.ps1
```

This creates `hf_deploy/` with all required files.

---

## Step 4: Set Up Git LFS (for large PDF files)

Hugging Face rejects files over 10MB. Use Git LFS for PDFs:

```powershell
cd hf_deploy

# Install Git LFS (run once; may need to install from git-lfs.com first)
git lfs install

# Track PDF files
git lfs track "*.pdf"

# Add the .gitattributes file LFS created
git add .gitattributes
```

---

## Step 5: Initialize Git and Commit

```powershell
# Initialize repo (if not already done)
git init

# Add all files
git add -A

# Commit
git commit -m "Add Paper Explorer app"
```

---

## Step 6: Connect to Hugging Face and Push

```powershell
# Add your Space as remote (replace YOUR_USERNAME and YOUR_SPACE_NAME)
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push (use token when prompted for password)
git push -u origin HEAD
```

**When prompted:**
- **Username:** your Hugging Face username
- **Password:** paste your token (not your account password)

**Or** embed the token in the URL (use a new token, don't reuse exposed ones):

```powershell
git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push -u origin HEAD
```

---

## Step 7: Wait for the Build

1. Open your Space: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Click the **Build logs** tab
3. Wait 5–15 minutes for the Docker image to build
4. Build steps:
   - Installing dependencies (numpy, matplotlib, marimo, qutip, etc.)
   - Copying app files
   - Starting Marimo server

---

## Step 8: Access Your App

When the build shows **Running** or **Built**:

1. Click **App** tab (or the main Space view)
2. Your Marimo Paper Explorer is live at the Space URL
3. Share: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

---

## Step 9: Verify It Works

Check that:

- [ ] Page loads without errors
- [ ] Introduction and sections render
- [ ] PDF figures display (Fig. 1, Fig. 2, etc.)
- [ ] Sliders and dropdowns work (SAFE-GRAPE, Filter Function, etc.)
- [ ] Citation links are blue and clickable
- [ ] Background has the warm cream color

---

## Troubleshooting

### Build fails

- Check build logs for the error
- Common: missing `app_port: 8080` in README.md → ensure `README_HF_SPACE.md` was copied as `README.md`
- Common: COPY fails → ensure all folders exist in `hf_deploy/`

### "Files over 10MB" error

- Run `git lfs install` and `git lfs track "*.pdf"`
- Add `.gitattributes`, then `git add -A` and commit again
- Push again

### App loads but shows errors

- Check browser console (F12) for JavaScript errors
- Some figures may need CSV data; ensure `Temperature and T2 simulations/` and `Entanglement simulations/` were included

### Authentication failed

- Create a new token at huggingface.co/settings/tokens
- Use token as password (not your HF account password)
- Ensure token has **Write** permission

---

## Updating the Deployed App

After making changes locally:

```powershell
cd c:\Users\anand43\Documents\Programmable_Quantum_Matter
.\prepare_hf_deploy.ps1
cd hf_deploy
git add -A
git commit -m "Update Paper Explorer"
git push
```

HF will automatically rebuild and redeploy.

---

## Summary Checklist

- [ ] Created HF Space (Docker, marimo template)
- [ ] Created HF token (Write)
- [ ] Ran `prepare_hf_deploy.ps1`
- [ ] Set up Git LFS for PDFs
- [ ] `git init`, `git add -A`, `git commit`
- [ ] `git remote add origin` (Space URL)
- [ ] `git push` (with token)
- [ ] Waited for build
- [ ] Opened Space URL and verified app
