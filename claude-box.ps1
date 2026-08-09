# claude-box.ps1 — run Claude Code fully autonomous, sandboxed to THIS project only.
# Usage:  .\claude-box.ps1

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

# 1. Build the image (fast after the first time — Docker caches the layers)
docker build -t claude-box "$projectRoot\.devcontainer"

# 2. Launch. Only this folder is mounted in; the rest of your PC is invisible.
#    A named volume keeps your Claude login so you don't re-auth every run.
docker run -it --rm `
    -v "${projectRoot}:/work" `
    -v "claude-box-config:/home/node/.claude" `
    -e "HOME=/home/node" `
    -w /work `
    claude-box `
    claude --dangerously-skip-permissions
