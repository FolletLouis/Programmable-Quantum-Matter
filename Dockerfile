# Programmable Quantum Matter — Paper Explorer
# Deploy: docker build -t pqm-explorer . && docker run -p 8080:8080 pqm-explorer

# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN uv pip install -r requirements.txt

# Application
COPY paper_explorer.py .
COPY custom.css .
COPY app/ app/
COPY PQM_latex_file/figs/ PQM_latex_file/figs/

# Precomputed CSV data (for interactive T2/entanglement figures)
COPY ["Temperature and T2 simulations", "Temperature and T2 simulations"]
COPY ["Entanglement simulations", "Entanglement simulations"]

EXPOSE 7860

RUN useradd -m app_user
USER app_user

CMD ["marimo", "run", "paper_explorer.py", "--host", "0.0.0.0", "-p", "7860"]
