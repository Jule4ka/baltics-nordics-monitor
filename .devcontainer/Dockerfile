FROM node:22-bookworm

# System deps: Python + git for your scraper project
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Python packages this repo uses (mirrors .github/workflows)
RUN pip3 install --break-system-packages --no-cache-dir \
        jupyter lxml pandas requests beautifulsoup4 html5lib scikit-learn

WORKDIR /work
CMD ["bash"]
