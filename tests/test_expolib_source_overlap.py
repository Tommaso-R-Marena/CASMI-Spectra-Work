"""Guardrails for source-overlap analyses of independent ExpoLib validation."""
import importlib.util
from pathlib import Path
import pytest

FILE=Path(__file__).resolve().parents[1]/"scripts"/"audit_expolib_source_overlap.py"
spec=importlib.util.spec_from_file_location("audit_expolib_source_overlap",FILE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_exact_overlap_audit_never_rescores_or_silently_resizes_source():
    with pytest.raises(ValueError,match="original full ExpoLib cohort changed"):
        m.analyze([],[],set(),set(),{})
