import math

from features.entropy import shannon_entropy


def test_empty_string_has_zero_entropy():
    assert shannon_entropy("") == 0.0


def test_single_symbol_has_zero_entropy():
    assert shannon_entropy("aaaa") == 0.0


def test_two_uniform_symbols():
    assert abs(shannon_entropy("ab") - 1.0) < 1e-9


def test_repetition_does_not_change_entropy():
    assert abs(shannon_entropy("abababab") - 1.0) < 1e-9


def test_four_uniform_symbols():
    assert abs(shannon_entropy("abcd") - 2.0) < 1e-9


def test_entropy_bounded_by_log2_alphabet_size():
    s = "example.com"
    assert 0.0 <= shannon_entropy(s) <= math.log2(len(set(s))) + 1e-9
