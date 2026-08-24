import json

from hac.polar import sha256_file
from tools.check_vcoco_v3_readiness import read_status


def test_readiness_rejects_source_drift(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"version": 1}\n', encoding="utf-8")
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "source_sha256": {"source": sha256_file(source)},
            }
        ),
        encoding="utf-8",
    )

    accepted = read_status(lock, {"LOCKED"}, {"source": source})
    assert accepted["accepted"]
    assert accepted["source_integrity"]["passed"]

    source.write_text('{"version": 2}\n', encoding="utf-8")
    drifted = read_status(lock, {"LOCKED"}, {"source": source})
    assert not drifted["accepted"]
    assert not drifted["source_integrity"]["passed"]
