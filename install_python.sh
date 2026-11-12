#!/usr/bin/env bash
# Create a local virtual environment (.venv) and install requirements.txt.
# Usage (from the repository root, in Git Bash / Linux / macOS):
#     bash install_python.sh
set -e

cd "$(dirname "$0")"

# python3 on Linux/macOS, python (or py) on Windows
if command -v python3 >/dev/null 2>&1 && python3 -c "" >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    PYTHON=py
fi

"$PYTHON" -m venv .venv

# The activation script is in Scripts/ on Windows, bin/ elsewhere
if [ -f .venv/Scripts/activate ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "Done. Activate the environment with:"
if [ -f .venv/Scripts/activate ]; then
    echo "    source .venv/Scripts/activate   (Git Bash)"
    echo "    .venv\Scripts\Activate.ps1      (PowerShell)"
else
    echo "    source .venv/bin/activate"
fi
