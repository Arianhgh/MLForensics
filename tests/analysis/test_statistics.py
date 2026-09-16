import math

from mlforensics.analysis import (
    bootstrap,
    effect_size,
    noninferiority_decision,
    paired_bootstrap,
    paired_differences,
    regression_decision,
)


def test_bootstrap_is_deterministic_with_explicit_seed():
    first = bootstrap([1, 2, 4, 8], seed=17, n_resamples=300)
    second = bootstrap([1, 2, 4, 8], seed=17, n_resamples=300)
    assert first == second
    assert first.lower <= first.estimate <= first.upper


def test_paired_differences_drop_failed_pairs_and_support_strict_mode():
    assert paired_differences([1, None, 3, float("nan")], [2, 4, None, 8]) == [1.0]
    assert paired_differences([1, 2], [2], strict=False) == [1.0]
    try:
        paired_differences([1, 2], [2], strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError("strict pairing should reject unequal lengths")


def test_paired_bootstrap_and_effect_size_have_expected_direction():
    result = paired_bootstrap([1, 2, 3], [2, 4, 6], n_resamples=100, seed=2)
    assert result.estimate > 0
    assert effect_size([2, 4, 6], baseline=[1, 2, 3]) > 0
    assert math.isinf(effect_size([2, 2], baseline=[1, 1]))


def test_practical_threshold_and_noninferiority_are_explicit():
    assert regression_decision(-0.2, (-0.3, -0.1), practical_threshold=0.05) == (True, False, False)
    assert noninferiority_decision((-0.02, 0.01), margin=0.05)
