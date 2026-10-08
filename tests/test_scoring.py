import pytest

from memshift.scoring.normalize import contract_valid, values_match


@pytest.mark.parametrize(
    ("vtype", "answer", "gold", "ok"),
    [
        ("org", "Northwind Labs", "Northwind Labs", True),
        ("org", "at Northwind Labs.", "Northwind Labs", True),
        ("org", "Northwind", "Northwind Labs", False),
        ("person", "Dana Okafor", "Dana Okafor", True),
        ("city", "Berlin, Germany", "Berlin", True),
        ("tz", "America/Chicago", "America/Chicago", True),
        ("tz", "Central time", "America/Chicago", False),
        ("diet", "She is fully vegan", "vegan", True),
        ("diet", "vegetarian", "vegan", False),
        ("commute", "biking", "bike", True),
        ("weekly", "Tuesdays at 10:30", "Tuesday 10:30", True),
        ("weekly", "Tuesday 10:00", "Tuesday 10:30", False),
        ("weekly", "Tuesday 2:30 PM", "Tuesday 14:30", True),
        ("datetime", "2026-10-22 16:15", "2026-10-22 16:15", True),
        ("datetime", "Thursday, October 22, 2026 at 16:15", "2026-10-22 16:15", True),
        ("datetime", "2026-10-22 16:30", "2026-10-22 16:15", False),
        ("date", "October 9, 2026", "2026-10-09", True),
        ("timerange", "09:00-17:00", "09:00-17:00", True),
        ("timerange", "9:00 to 17:00", "09:00-17:00", True),
        ("set_person", ["Ann Lee", "Bo Chen"], ["Bo Chen", "Ann Lee"], True),
        ("set_person", "Ann Lee and Bo Chen", ["Bo Chen", "Ann Lee"], True),
        ("set_person", ["Ann Lee"], ["Bo Chen", "Ann Lee"], False),
        ("set_person", [], [], True),
        ("set_person", "none", [], True),
    ],
)
def test_values_match(vtype, answer, gold, ok):
    assert values_match(vtype, answer, gold) is ok


def test_contract_valid_requires_status():
    assert contract_valid("org", "abstain", None, "abstain", None)
    assert not contract_valid("org", "answer", "X", "abstain", None)
    assert not contract_valid("org", "abstain", None, "answer", "X")
    assert contract_valid("org", "blocked", None, "blocked", "X")
