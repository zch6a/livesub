"""The CI workflows: every action pinned to a commit, no token left in the checkout, read-only by default."""

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def test_workflows_are_pinned_and_read_only():
    files = sorted(WORKFLOWS.glob("*.yml"))
    assert files
    for f in files:
        yml = f.read_text(encoding="utf-8")
        uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)", yml, flags=re.M)
        assert uses, f.name
        for u in uses:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u), f"{f.name}: {u}"
        for block in yml.split("uses: actions/checkout@")[1:]:
            assert re.match(r"\S+[^\n]*\n\s+with:\n\s+persist-credentials: false", block), f"{f.name}: checkout keeps the token"
        assert re.search(r"^permissions:\n\s+contents: read$", yml, flags=re.M), f"{f.name}: read-only by default"
