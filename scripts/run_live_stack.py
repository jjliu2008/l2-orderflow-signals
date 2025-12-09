"""
Convenience launcher to start the live data recorder and the PyQtGraph viewer together.

Usage:
  python scripts/run_live_stack.py

Environment overrides:
  COINBASE_*          # for run_build_dataset (JWT, URL, product IDs, etc.)
  WS_OUTPUT_DIR       # defaults to data/live
  PYQT_DATA_DIR       # defaults to data/live
  REFRESH_MS          # defaults to 1000
  MC_REFRESH_MS       # defaults to 30000
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    py = sys.executable

    processes = []
    commands = [
        ("recorder", [py, str(project_root / "scripts" / "run_build_dataset.py")]),
        # Swap Dash for PyQtGraph viewer
        ("pyqt_viewer", [py, str(project_root / "scripts" / "run_pyqt_viewer.py")]),
    ]

    try:
        for name, cmd in commands:
            print(f"Starting {name}: {' '.join(cmd)}")
            env = os.environ.copy()
            # Preserve interactive options for the viewer; set SHOW_* or PYQT_NO_PROMPT in your shell if desired.
            proc = subprocess.Popen(cmd, cwd=project_root, env=env)
            processes.append((name, proc))

        # Wait on the viewer process; keep the recorder running in background.
        viewer_proc = next(p for n, p in processes if n == "pyqt_viewer")
        viewer_proc.wait()
    except KeyboardInterrupt:
        print("Received interrupt, shutting down...")
    finally:
        for name, proc in processes:
            if proc.poll() is None:
                print(f"Terminating {name} (pid {proc.pid})")
                proc.terminate()
        for _, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
