from mlforensics.core import FailureSignature
from mlforensics.diagnose.bisect import BisectResult, RunOutcome
from mlforensics.diagnose.report import DiagnosisReport, as_dict, to_json


def test_report_is_structured_and_json_stable():
    result = BisectResult("bad", "good", "bad", [RunOutcome("bad", False, score=2)])
    report = DiagnosisReport("bisect", "regression", evidence={"result": result})
    data = as_dict(report)
    assert data["evidence"]["result"]["first_bad"] == "bad"
    assert '"kind": "bisect"' in to_json(report)


def test_report_keeps_core_evidence_wire_type_markers():
    signature = FailureSignature.structured("timeout", message="hung")
    report = DiagnosisReport("shrink", "inconclusive", evidence={"failure": signature})
    assert as_dict(report)["evidence"]["failure"]["type"] == "failure_signature"
