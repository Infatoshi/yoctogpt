from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v1_script_runs_end_to_end() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "v1.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    assert "step 0: val" in result.stdout
    assert "step 500: val" in result.stdout
    assert "Peter Piper " in result.stdout
