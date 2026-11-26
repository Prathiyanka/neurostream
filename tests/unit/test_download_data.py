import pytest
from download_data import (
    RecordingInfo,
    SubjectReport,
    check_recording,
    parse_subjects,
    summarize,
)


def good_info(**overrides: object) -> RecordingInfo:
    base: dict[str, object] = {
        "sfreq": 160.0,
        "n_channels": 64,
        "duration_s": 125.0,
        "annotations": ["T0", "T1", "T0", "T2"] * 8,
    }
    base.update(overrides)
    return RecordingInfo(**base)  # type: ignore[arg-type]


def test_good_recording_has_no_issues() -> None:
    assert check_recording(good_info()) == []


def test_wrong_sampling_rate_is_flagged() -> None:
    issues = check_recording(good_info(sfreq=128.0))
    assert any("sampling rate 128" in i for i in issues)


def test_wrong_channel_count_is_flagged() -> None:
    assert any("channels" in i for i in check_recording(good_info(n_channels=63)))


def test_short_recording_is_flagged() -> None:
    assert any("only 30.0 s" in i for i in check_recording(good_info(duration_s=30.0)))


def test_missing_and_unknown_labels_are_flagged() -> None:
    issues = check_recording(good_info(annotations=["T0", "T1", "X9"] * 10))
    assert any("unexpected" in i and "X9" in i for i in issues)
    assert any("missing" in i and "T2" in i for i in issues)


def test_too_few_trials_is_flagged() -> None:
    issues = check_recording(good_info(annotations=["T0", "T1", "T2", "T0"]))
    assert any("imagery trials" in i for i in issues)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("1-3", [1, 2, 3]), ("5,1,3-4", [1, 3, 4, 5]), ("2,2,2", [2]), ("1-109", list(range(1, 110)))],
)
def test_parse_subjects(spec: str, expected: list[int]) -> None:
    assert parse_subjects(spec) == expected


@pytest.mark.parametrize("spec", ["0", "110", "5-3", "abc"])
def test_parse_subjects_rejects_bad_input(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_subjects(spec)


def test_summarize_reports_disagreement_with_common_list() -> None:
    reports = [
        SubjectReport(subject=1, usable=True),
        SubjectReport(subject=7, usable=False, issues=["x"]),
        SubjectReport(subject=88, usable=False, issues=["x"]),
        SubjectReport(subject=92, usable=True),
    ]
    s = summarize(reports)
    assert s["excluded"] == [7, 88]
    assert s["excluded_but_not_commonly"] == [7]
    assert s["commonly_excluded_but_passed"] == [92]
