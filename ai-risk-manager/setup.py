"""
One-command project setup.
Run: python setup.py
"""
import subprocess, sys, os
from pathlib import Path

PYTHON = sys.executable
ROOT   = Path(__file__).parent
os.chdir(ROOT)
env = {**os.environ, "PYTHONPATH": str(ROOT)}


def run(cmd, **kwargs):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print("ERROR — aborting."); sys.exit(1)


def main():
    print("=" * 50)
    print("  Sentinel — Setup")
    print("=" * 50)

    # 1. Install Python deps
    run([PYTHON, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])

    # 2. Generate data
    run([PYTHON, "-m", "ml.data_generator"], env=env)

    # 3. Train models
    run([PYTHON, "-m", "ml.train"], env=env)

    # 4. Evaluate
    run([PYTHON, "-m", "ml.evaluate"], env=env)

    # 5. Copy .env
    if not Path(".env").exists():
        Path(".env").write_text(Path(".env.example").read_text())
        print("\n>>> Created .env from .env.example")

    print("\n" + "=" * 50)
    print("  Setup complete!")
    print("=" * 50)
    print("\nTo start the backend:")
    print(f"  cd {ROOT}")
    print(f"  PYTHONPATH=. {PYTHON} -m uvicorn backend.main:app --reload")
    print("\nTo start the frontend (separate terminal):")
    print("  cd frontend && npm install && npm run dev")
    print("\nAPI docs: http://localhost:8000/docs")
    print("Dashboard: http://localhost:3000")


if __name__ == "__main__":
    main()
