# Prepare files for Hugging Face Space deployment
# Run from repo root: .\prepare_hf_deploy.ps1

$deployDir = "hf_deploy"
if (Test-Path $deployDir) { Remove-Item -Recurse -Force $deployDir }
New-Item -ItemType Directory -Path $deployDir | Out-Null

# Core files
Copy-Item "Dockerfile" $deployDir
Copy-Item "requirements.txt" $deployDir
Copy-Item "paper_explorer.py" $deployDir
Copy-Item "custom.css" $deployDir

# README for HF (with app_port: 8080)
Copy-Item "README_HF_SPACE.md" "$deployDir\README.md"

# App package
Copy-Item -Recurse "app" "$deployDir\app"

# Figures
New-Item -ItemType Directory -Path "$deployDir\PQM_latex_file\figs" -Force | Out-Null
Copy-Item "PQM_latex_file\figs\*" "$deployDir\PQM_latex_file\figs\"

# Data folders
Copy-Item -Recurse "Temperature and T2 simulations" "$deployDir\"
Copy-Item -Recurse "Entanglement simulations" "$deployDir\"

Write-Host "Done! Files copied to $deployDir\" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Create Space at huggingface.co/spaces (Docker, marimo or Blank template)"
Write-Host "2. cd $deployDir"
Write-Host "3. git init"
Write-Host "4. git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME"
Write-Host "5. git add ."
Write-Host "6. git commit -m 'Add Paper Explorer'"
Write-Host "7. git push -u origin main"
