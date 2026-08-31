"""Script to launch the NFL Gridiron AI Predictor Web Dashboard."""

import sys
from pathlib import Path
import uvicorn

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    port = 8000
    print(f"🏈 Starting NFL Gridiron AI Predictor Dashboard at http://127.0.0.1:{port}")
    uvicorn.run("src.web.app:app", host="127.0.0.1", port=port, log_level="info")
