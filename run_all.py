from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a two-epoch smoke test.")
    args = parser.parse_args()
    start = time.time()
    # Dataset has already been prepared; validate the available test suite before training.
    test_dir = ROOT / "tests"
    test_target = str(test_dir) if test_dir.exists() else "tests"
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", test_target], cwd=ROOT)
    if result.returncode:
        raise SystemExit("Protocol tests failed; fix them before training.")
    cmd = [sys.executable, "train_model.py"]
    if args.quick: cmd.append("--quick")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode:
        raise SystemExit("Training/evaluation failed. Review the traceback above.")
    print(f"Pipeline completed in {time.time()-start:.1f} seconds.")
    print("Start UI with: python -m streamlit run app.py")
if __name__ == "__main__":
    main()
