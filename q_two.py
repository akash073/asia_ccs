"""Run the YOLO26n and quantum training scripts sequentially.

Place this launcher in the same directory as both scripts, then run:
    python run_two_scripts.py

The second job starts only if the first exits successfully.
"""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "quantum_yolo.py",
    "quantum_dnn_cnn.py",
]


def main():
    for index, filename in enumerate(SCRIPTS, start=1):
        script = HERE / filename
        if not script.is_file():
            raise FileNotFoundError(f"Missing script: {script}")
        print(f"\n[{index}/{len(SCRIPTS)}] Running {filename}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=str(HERE), check=True)
        print(f"[{index}/{len(SCRIPTS)}] Finished {filename}", flush=True)
    print("\nBoth scripts completed successfully.", flush=True)


if __name__ == "__main__":
    main()
