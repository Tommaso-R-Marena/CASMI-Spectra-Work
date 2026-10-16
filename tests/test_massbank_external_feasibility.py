import importlib.util
from pathlib import Path

MODULE=Path(__file__).resolve().parents[1]/"scripts"/"audit_massbank_external_energy_geometry.py"
spec=importlib.util.spec_from_file_location("audit_massbank_external_energy_geometry",MODULE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def record(*, energy="20 eV", key="UHOVQNZJYSORNB-UHFFFAOYSA-N", instrument="QToF", frag="CID"):
    return "\n".join([
        "ACCESSION: TEST001",
        "CH$FORMULA: C8H10N4O2",
        f"CH$LINK: INCHIKEY {key}",
        f"AC$INSTRUMENT: {instrument}",
        "AC$INSTRUMENT_TYPE: LC-ESI-QTOF",
        "AC$MASS_SPECTROMETRY: MS_TYPE MS2",
        "AC$MASS_SPECTROMETRY: ION_MODE POSITIVE",
        f"AC$MASS_SPECTROMETRY: FRAGMENTATION_MODE {frag}",
        f"AC$MASS_SPECTROMETRY: COLLISION_ENERGY {energy}",
        "MS$FOCUSED_ION: PRECURSOR_TYPE [M+H]+",
        "PK$NUM_PEAK: 8",
        "PK$PEAK: m/z int. rel.int.",
        "100.0 5 100",
        "//",
    ])


def test_massbank_parser_accepts_only_explicit_single_ev():
    name="sample/record/ABC/TEST001.txt"
    got=m.parse_record_metadata(record(),name)
    assert got["eV"]==20.0 and got["contributor"]=="ABC"
    for bad in ("20","20 NCE","20 %","Ramp 10-50 eV","20,40 eV","15 V"):
        assert m.parse_record_metadata(record(energy=bad),name) is None


def test_massbank_parser_rejects_missing_identity_and_unknown_instrument():
    path="sample/record/ABC/TEST001.txt"
    assert m.parse_record_metadata(record(key="bad"),path) is None
    assert m.parse_record_metadata(record(instrument=""),path) is None
    assert m.parse_record_metadata(record(), "TEST001.txt") is None
    assert m.parse_record_metadata(record(), "record/TEST001.txt") is None


def test_massbank_parser_accepts_official_release_contributor_layout():
    # Official MassBank 2026.03 repository and release use contributor dirs.
    for path in (
        "AAFC/MSBNK-AAFC-AC000001.txt",
        "MassBank-data-2026.03/AAFC/MSBNK-AAFC-AC000001.txt",
        "MassBank-data-2026.03/record/AAFC/MSBNK-AAFC-AC000001.txt",
    ):
        got=m.parse_record_metadata(record(),path)
        assert got is not None
        assert got["contributor"]=="AAFC"
    # Packaging correction never licenses ambiguous or uncalibrated energies.
    for bad_energy in ("10 NCE", "20 V", "20 %", "10-40 eV", "20"):
        assert m.parse_record_metadata(
            record(energy=bad_energy),
            "MassBank-data-2026.03/AAFC/MSBNK-AAFC-AC000001.txt",
        ) is None


def test_eligible_triplet_is_source_and_instrument_consistent():
    base=m.parse_record_metadata(record(),"sample/record/ABC/TEST001.txt")
    records=[]
    for e in (10,20,40):
        r=dict(base)
        r["eV"]=float(e)
        r["accession"]=f"TEST{e:03d}"
        r["path"]=f"sample/record/ABC/TEST{e:03d}.txt"
        records.append(r)
    rows,counts=m.eligible_triplets(records,excluded=set())
    assert len(rows)==1
    assert rows[0]["low_eV"]==10
    assert rows[0]["high_eV"]==40
    assert abs(rows[0]["t"]-1/3)<1e-12
    assert m.eligible_triplets(records,excluded={base["inchikey"]})[0]==[]
    dup=records+[dict(records[1])]
    assert m.eligible_triplets(dup,excluded=set())[0]==[]
