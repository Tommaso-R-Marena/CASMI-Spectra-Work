from pathlib import Path
import subprocess
import sys


def test_msnlib_variants_cli_imports_and_shows_help():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_msnlib_variants.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Open MSnLib censor-aware and mixture-geometry variants" in proc.stdout
