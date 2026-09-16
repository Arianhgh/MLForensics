import math

from mlforensics.core import Run, RunCapsule, TraceEvent
from mlforensics.diagnose.trace import TensorTracer, TraceBuffer, trace_incident


class Array:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def tolist(self):
        return self.values


def test_trace_builds_causal_path_and_reports_eviction() -> None:
    tracer = TensorTracer(max_events=2)
    source = Array([1.0])
    middle = Array([2.0])
    bad = Array([math.nan])
    first = tracer.record("source", source, source="loader")
    second = tracer.record("middle", middle, inputs=source)
    abnormal = tracer.record("bad", bad, inputs=middle)

    assert first is not None and second is not None and abnormal is not None
    assert list(abnormal.data["parents"]) == [second.tensor_id]
    analysis = tracer.buffer.analyze()
    assert analysis["dropped_events"] == 1
    assert [item["kind"] for item in analysis["causal_path"]] == ["middle", "bad"]
    assert analysis["missing_parents"] == [first.tensor_id]


def test_first_abnormal_is_preserved_when_ring_buffer_evicts_it() -> None:
    trace = TraceBuffer(1)
    abnormal = trace.record_tensor("bad", [math.inf])
    trace.record_tensor("later", [1.0])
    assert trace.first_abnormal() == abnormal
    assert trace.dropped_events == 1


def test_trace_incident_reads_structured_capsule_evidence() -> None:
    trace = TraceBuffer(4)
    trace.record_tensor("source", [1.0], tensor_id="source")
    trace.record_tensor("bad", [math.nan], tensor_id="bad", parents=("source",))
    capsule = RunCapsule(
        Run(run_id="trace-evidence", status="failed", started_at="t"),
        evidence={"tensor_trace": trace.to_dict()},
    )

    analysis = trace_incident(capsule)

    assert analysis["first_abnormal"]["kind"] == "bad"
    assert [event["kind"] for event in analysis["causal_path"]] == ["source", "bad"]


def test_tensor_evidence_survives_a_run_that_also_logged_events() -> None:
    """A realistic run logs its own events, which must not hide the tensor trace."""
    trace = TraceBuffer(4)
    trace.record_tensor("source", [1.0], tensor_id="source")
    trace.record_tensor("bad", [math.nan], tensor_id="bad", parents=("source",))
    capsule = RunCapsule(
        Run(
            run_id="trace-with-events",
            status="failed",
            started_at="t",
            events=(TraceEvent("nan_detected", data={"step": 3}),),
        ),
        evidence={"tensor_trace": trace.to_dict()},
    )

    analysis = trace_incident(capsule)

    assert analysis["first_abnormal"]["kind"] == "bad"
    assert [event["kind"] for event in analysis["causal_path"]] == ["source", "bad"]
    assert "nan_detected" in [event["kind"] for event in analysis["events"]]


def test_recorded_events_are_not_duplicated_by_the_merge() -> None:
    trace = TraceBuffer(2)
    event = trace.record_tensor("bad", [math.nan], tensor_id="bad")
    capsule = RunCapsule(
        Run(run_id="trace-duplicate", status="failed", started_at="t", events=(event,)),
        evidence={"tensor_trace": trace.to_dict()},
    )

    assert len(trace_incident(capsule)["events"]) == 1


def test_tensor_tracer_records_the_training_step() -> None:
    tracer = TensorTracer(max_events=4)
    recorded = tracer.record("logits", Array([1.0]), step=7)

    assert recorded is not None
    assert recorded.step == 7
    assert [item.step for item in tracer.events()] == [7]
