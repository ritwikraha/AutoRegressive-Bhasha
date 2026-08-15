from ocn.detectors import OCNDetector, approximate_token_count


def test_detects_not_just_but() -> None:
    result = OCNDetector().detect("Leadership is not just about leading people; it is about guiding them.")
    assert result.has_ocn
    assert result.count == 1


def test_detects_more_than_just() -> None:
    result = OCNDetector().detect("The app is more than just a tracker. It changes habits.")
    assert result.has_ocn


def test_ignores_plain_negation() -> None:
    result = OCNDetector().detect("A virus is not a cell.")
    assert not result.has_ocn


def test_approximate_token_count() -> None:
    assert approximate_token_count("Not just X, but Y.") == 7
