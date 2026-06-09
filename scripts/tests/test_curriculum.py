from scripts.curriculum import SECTIONS, Section, Topic


def test_thirteen_sections_numbered_1_to_13():
    assert len(SECTIONS) == 13
    assert [s.number for s in SECTIONS] == list(range(1, 14))


def test_section_and_topic_slugs_are_unique_within_section():
    for s in SECTIONS:
        slugs = [t.slug for t in s.topics]
        assert len(slugs) == len(set(slugs)), f"dupe topic slug in {s.slug}"


def test_machine_learning_is_section_4_with_gradient_descent_first():
    ml = next(s for s in SECTIONS if s.number == 4)
    assert ml.slug == "machine-learning"
    assert ml.topics[0].slug == "gradient-descent"


def test_types():
    assert all(isinstance(s, Section) for s in SECTIONS)
    assert all(isinstance(t, Topic) for s in SECTIONS for t in s.topics)
