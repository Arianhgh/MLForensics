from pathlib import Path

import pytest

from mlforensics.core.codecs import (
    CONTAINER_CODEC,
    STATE_TREE_KEY,
    decode_state_tree,
    default_state_codecs,
    encode_state_tree,
)


def test_typed_mapping_keys_and_tuples_round_trip():
    original = {0: (1, 2), 1: {"nested": (3, 4)}, "ok": [5, 6]}
    encoded = encode_state_tree(original)
    assert encoded[STATE_TREE_KEY] == 1
    assert encoded["codec"] == CONTAINER_CODEC
    assert encoded["kind"] == "mapping"
    decoded = decode_state_tree(encoded)
    assert decoded == original
    assert isinstance(decoded[0], tuple)
    assert list(decoded) == [0, 1, "ok"]


def test_bytes_and_unsupported_state(tmp_path: Path):
    payloads: dict[str, bytes] = {}

    def store(payload: bytes, metadata):
        digest = str(len(payloads))
        payloads[digest] = payload
        return digest

    encoded = encode_state_tree({"blob": b"abc"}, store)
    decoded = decode_state_tree(encoded, payloads.__getitem__)
    assert decoded["blob"] == b"abc"

    class NotPortable:
        pass

    with pytest.raises(TypeError, match="not replayable"):
        encode_state_tree(NotPortable())


def test_numpy_empty_and_scalar_round_trip():
    np = pytest.importorskip("numpy")
    payloads: dict[str, bytes] = {}

    def store(payload: bytes, metadata):
        digest = metadata.get("dtype", "x") + str(len(payloads))
        payloads[digest] = payload
        return digest

    values = [np.array(3.5), np.empty((0, 3), dtype=np.float32)]
    encoded = encode_state_tree(values, store)
    decoded = decode_state_tree(encoded, payloads.__getitem__)
    assert decoded[0].shape == ()
    assert decoded[0].dtype == np.dtype("float64")
    assert float(decoded[0]) == 3.5
    assert decoded[1].shape == (0, 3)
    assert decoded[1].dtype == np.float32


def test_torch_scalar_empty_and_adam_state_round_trip():
    torch = pytest.importorskip("torch")
    payloads: dict[str, bytes] = {}

    def store(payload: bytes, metadata):
        digest = f"{metadata.get('dtype')}-{len(payloads)}"
        payloads[digest] = payload
        return digest

    scalar = torch.tensor(1.25)
    empty = torch.empty(0, 4, dtype=torch.float32)
    encoded = encode_state_tree({"scalar": scalar, "empty": empty}, store)
    decoded = decode_state_tree(encoded, payloads.__getitem__)
    assert decoded["scalar"].shape == ()
    assert decoded["scalar"].dtype == torch.float32
    assert abs(float(decoded["scalar"]) - 1.25) < 1e-6
    assert tuple(decoded["empty"].shape) == (0, 4)

    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = model(torch.ones(3, 2)).sum()
    loss.backward()
    optimizer.step()
    captured = encode_state_tree(optimizer.state_dict(), store)
    restored = decode_state_tree(captured, payloads.__getitem__)
    assert all(isinstance(key, int) for key in restored["state"])
    step = restored["state"][0]["step"]
    assert torch.is_tensor(step) or int(step) >= 1
    clone = torch.nn.Linear(2, 2, bias=False)
    clone.load_state_dict(model.state_dict())
    clone_opt = torch.optim.Adam(clone.parameters(), lr=0.01)
    clone_opt.load_state_dict(restored)
    assert (
        clone_opt.state_dict()["state"][0]["exp_avg"].shape
        == optimizer.state_dict()["state"][0]["exp_avg"].shape
    )


def test_json_codec_does_not_claim_tensors():
    torch = pytest.importorskip("torch")
    assert not any(
        codec.name == "json" and codec.can_encode(torch.tensor([1.0]))
        for codec in default_state_codecs.codecs
        if codec.name == "json"
    )


def test_input_digest_hashes_full_tensor_contents_not_printed_text():
    torch = pytest.importorskip("torch")
    from mlforensics.core.codecs import digest_state_tree

    left = torch.zeros(2000)
    right = torch.zeros(2000)
    right[100] = 1.0
    assert str(left) == str(right)
    first = digest_state_tree(left)
    second = digest_state_tree(right)
    assert first is not None and second is not None
    assert first != second
    assert digest_state_tree(torch.zeros(2000)) == first


def test_user_dictionaries_that_resemble_codec_markers_round_trip():
    original = {"codec": CONTAINER_CODEC, "kind": "tuple", "items": [1, 2]}
    decoded = decode_state_tree(encode_state_tree(original))
    assert decoded == original
    assert isinstance(decoded, dict)

    legacy_tuple = {"codec": CONTAINER_CODEC, "kind": "tuple", "items": [1, 2]}
    assert decode_state_tree(legacy_tuple) == (1, 2)
