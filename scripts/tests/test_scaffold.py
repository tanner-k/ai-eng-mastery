from pathlib import Path

from scripts.curriculum import SECTIONS
from scripts.scaffold import generate


def test_generate_creates_all_sections_and_topics(tmp_path: Path):
    generate(tmp_path)
    for section in SECTIONS:
        sec_dir = tmp_path / section.dirname
        assert (sec_dir / "README.md").exists()
        for i, topic in enumerate(section.topics, start=1):
            tdir = sec_dir / f"{i:02d}-{topic.slug}"
            assert (tdir / "README.md").exists()
            assert (tdir / "interview.md").exists()
            assert (tdir / "exercises.md").exists()
            assert (tdir / "mini_project" / "README.md").exists()
            assert (tdir / "solutions" / ".gitkeep").exists()


def test_topic_readme_has_title_substituted(tmp_path: Path):
    generate(tmp_path)
    readme = (tmp_path / "04-machine-learning" / "01-gradient-descent" / "README.md").read_text()
    assert "Gradient Descent" in readme
    assert "{{TOPIC_TITLE}}" not in readme
    interview = (tmp_path / "04-machine-learning" / "01-gradient-descent" / "interview.md").read_text()
    assert "Gradient Descent" in interview
    assert "{{TOPIC_TITLE}}" not in interview


def test_generate_writes_progress_table(tmp_path: Path):
    generate(tmp_path)
    progress = (tmp_path / "PROGRESS.md").read_text()
    assert "Gradient Descent" in progress
    assert "| Topic |" in progress


def test_generate_is_idempotent_and_preserves_existing(tmp_path: Path):
    generate(tmp_path)
    target = tmp_path / "04-machine-learning" / "01-gradient-descent" / "README.md"
    target.write_text("REAL CONTENT")
    generate(tmp_path)  # second run must not clobber
    assert target.read_text() == "REAL CONTENT"
