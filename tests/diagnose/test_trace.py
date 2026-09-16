import math

from mlforensics.diagnose.trace import TraceBuffer, tensor_event


def test_ring_buffer_evicts_oldest_but_keeps_first_abnormal():
    trace = TraceBuffer(2)
    trace.record_tensor("a", 1)
    trace.record(trace.record_tensor("bad", float("nan")))
    trace.record_tensor("c", 3)
    assert [e.kind for e in trace.events()] == ["bad", "c"]
    assert trace.first_abnormal_event().kind == "bad"
    assert trace.first_abnormal_event().data["source"] is None


def test_nested_nonfinite_detection():
    event = tensor_event("nested", [1, math.inf], source="dense")
    assert event.data["abnormal"]
    assert event.data["source"] == "dense"
