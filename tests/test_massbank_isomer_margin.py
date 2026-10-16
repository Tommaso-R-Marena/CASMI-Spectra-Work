"""Synthetic checks for frozen secondary isomer margin audit."""
import importlib.util
from pathlib import Path
import pytest

FILE=Path(__file__).resolve().parents[1]/"scripts"/"audit_massbank_isomer_margin.py"
spec=importlib.util.spec_from_file_location("audit_massbank_isomer_margin",FILE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_margin_audit_rejects_short_or_reused_query_population():
    row={"query_path":"x","formula":"F","alpha_quarter_margin":.1,
         "hellinger_fixed_margin":.2,"alpha_quarter_rr":.5,
         "hellinger_fixed_rr":.5,"contributor":"A","instrument_type":"QTOF"}
    with pytest.raises(ValueError,match="252"):
        m.audit([row])
