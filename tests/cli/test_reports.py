import json
from xml.etree import ElementTree

from mlforensics.ci import CICheck, CIResult
from mlforensics.cli.main import _ci_report


def _result() -> CIResult:
    return CIResult(
        False,
        (
            CICheck("healthy", True),
            CICheck("regression", False, "fail", {"metric": "accuracy"}),
            CICheck("uncertain", True, "warn", {"reason": "insufficient evidence"}),
        ),
    )


def test_ci_report_formats_are_machine_consumable() -> None:
    result = _result()

    junit = ElementTree.fromstring(_ci_report(result, "junit"))
    assert junit.tag == "testsuite"
    assert junit.attrib["failures"] == "1"
    assert len(junit.findall("testcase")) == 3

    sarif = json.loads(_ci_report(result, "sarif"))
    assert sarif["version"] == "2.1.0"
    assert sarif["runs"][0]["invocations"][0]["exitCode"] == 1
    assert sarif["runs"][0]["results"][1]["level"] == "error"

    checks = json.loads(_ci_report(result, "github"))
    assert checks["name"] == "MLForensics"
    assert checks["conclusion"] == "failure"
    assert checks["annotations"][0]["annotation_level"] == "failure"
