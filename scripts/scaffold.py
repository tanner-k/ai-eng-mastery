"""Generate section/topic scaffolding and PROGRESS.md from the curriculum manifest."""
from __future__ import annotations

from pathlib import Path

from scripts.curriculum import SECTIONS, Section, Topic

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "TOPIC_TEMPLATE"


def _write_if_absent(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` only if it does not already exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _render_template(rel: str, topic: Topic) -> str:
    raw = (_TEMPLATE_DIR / rel).read_text()
    return raw.replace("{{TOPIC_TITLE}}", topic.title)


def _topic_dir(root: Path, section: Section, index: int, topic: Topic) -> Path:
    return root / section.dirname / f"{index:02d}-{topic.slug}"


def _scaffold_topic(root: Path, section: Section, index: int, topic: Topic) -> None:
    tdir = _topic_dir(root, section, index, topic)
    _write_if_absent(tdir / "README.md", _render_template("README.md", topic))
    _write_if_absent(tdir / "interview.md", _render_template("interview.md", topic))
    _write_if_absent(tdir / "exercises.md", _render_template("exercises.md", topic))
    _write_if_absent(tdir / "solutions" / ".gitkeep", "")
    _write_if_absent(
        tdir / "mini_project" / "README.md",
        _render_template("mini_project/README.md", topic),
    )


def _section_readme(section: Section) -> str:
    lines = [f"# {section.number:02d} — {section.title}", "", "## Topics", ""]
    for i, topic in enumerate(section.topics, start=1):
        lines.append(f"{i}. [{topic.title}]({i:02d}-{topic.slug}/README.md)")
    lines.append("")
    return "\n".join(lines)


def _progress_table() -> str:
    lines = [
        "# Progress",
        "",
        "Status legend: ⬜ not started · 🟡 in progress · ✅ done",
        "",
        "| Topic | Notes | Impl | Interview | Exercises |",
        "| --- | --- | --- | --- | --- |",
    ]
    for section in SECTIONS:
        lines.append(f"| **{section.title}** | | | | |")
        for topic in section.topics:
            lines.append(f"| {topic.title} | ⬜ | ⬜ | ⬜ | ⬜ |")
    lines.append("")
    return "\n".join(lines)


def generate(root: Path) -> None:
    """Generate all section/topic stubs and PROGRESS.md under ``root``.

    Existing files are never overwritten, so real content survives re-runs.
    """
    root = Path(root)
    for section in SECTIONS:
        sec_dir = root / section.dirname
        _write_if_absent(sec_dir / "README.md", _section_readme(section))
        for i, topic in enumerate(section.topics, start=1):
            _scaffold_topic(root, section, i, topic)
    # PROGRESS.md is regenerated each run (it is a derived index, not hand-edited).
    (root / "PROGRESS.md").write_text(_progress_table())


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    generate(repo_root)
    print(f"Scaffold generated under {repo_root}")
