from pathlib import Path
import subprocess
import sys


def test_msnlib_group_validation_cli_imports_and_shows_help():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_msnlib_group_validation.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Formula-block bootstrap" in proc.stdout
