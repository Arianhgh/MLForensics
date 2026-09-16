from mlforensics.diagnose.bisect import BisectResult, RunOutcome
from mlforensics.diagnose.report import DiagnosisReport, as_dict, to_json


def test_report_is_structured_and_json_stable():
    result = BisectResult("bad", "good", "bad", [RunOutcome("bad", False, score=2)])
    report = DiagnosisReport("bisect", "regression", evidence={"result": result})
    data = as_dict(report)
    assert data["evidence"]["result"]["first_bad"] == "bad"
    assert '"kind": "bisect"' in to_json(report)
