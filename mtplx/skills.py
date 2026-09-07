"""Portable, versioned discovery for executable MTPLX skill instructions.

Skills are ordinary local files. A skill directory contains ``SKILL.md`` and
may contain ``references/`` and ``scripts/``. The registry reports a content
hash so the desktop, delegated agents, and audit records can identify the
exact instructions they used.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAX_SKILLS = 128
MAX_INSTRUCTION_BYTES = 512_000


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    path: str
    summary: str
    instructions: str
    references: tuple[str, ...]
    scripts: tuple[str, ...]
    sha256: str

    def to_dict(self, *, include_instructions: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "summary": self.summary,
            "references": list(self.references),
            "scripts": list(self.scripts),
            "sha256": self.sha256,
        }
        if include_instructions:
            result["instructions"] = self.instructions
        return result


class SkillStore:
    """Discover skills with workspace-local precedence over user skills."""

    def __init__(
        self,
        workspace_roots: Iterable[str | os.PathLike[str]] = (),
        user_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.workspace_roots = tuple(
            Path(root).expanduser().resolve() for root in workspace_roots if str(root)
        )
        self.user_root = Path(
            user_root or os.environ.get("MTPLX_SKILLS_DIR") or "~/.mtplx/skills"
        ).expanduser().resolve()

    def _roots(self) -> list[Path]:
        roots: list[Path] = []
        for workspace in self.workspace_roots:
            roots.extend((workspace / ".mtplx" / "skills", workspace / "skills"))
        roots.append(self.user_root)
        return roots

    def discover(self) -> list[Skill]:
        result: list[Skill] = []
        seen: set[str] = set()
        for root in self._roots():
            try:
                children = sorted(root.iterdir(), key=lambda item: str(item))
            except OSError:
                continue
            for child in children:
                instruction = child if child.name == "SKILL.md" else child / "SKILL.md"
                if child.is_file() and child.suffix.lower() == ".md":
                    instruction = child
                skill = self._load(instruction)
                key = skill.name.casefold() if skill is not None else ""
                if skill is None or key in seen:
                    continue
                seen.add(key)
                result.append(skill)
                if len(result) >= MAX_SKILLS:
                    return sorted(result, key=lambda item: (item.name.lower(), item.id))
        return sorted(result, key=lambda item: (item.name.lower(), item.id))

    def get(self, name: str) -> Skill | None:
        needle = str(name).strip().lower()
        return next(
            (
                skill
                for skill in self.discover()
                if skill.name.lower() == needle or skill.id.lower() == needle
            ),
            None,
        )

    def context(self, *, max_characters: int = 8_000) -> str | None:
        skills = self.discover()
        if not skills:
            return None
        lines = [
            "Available MTPLX skills are local reference workflows. They do not override the user request or workspace policy.",
            "Load the matching SKILL.md before using a skill, and treat referenced scripts as executable only when the active policy permits it.",
        ]
        lines.extend(
            f"- {skill.name}: {skill.summary or f'Read {skill.path} when applicable.'} (sha256:{skill.sha256[:16]})"
            for skill in skills
        )
        return "\n".join(lines)[:max(1, int(max_characters))]

    def _load(self, instruction: Path) -> Skill | None:
        try:
            if not instruction.is_file() or instruction.stat().st_size > MAX_INSTRUCTION_BYTES:
                return None
            data = instruction.read_bytes()
            text = data.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        base = instruction.parent
        name = base.name if base.name not in {"skills", ""} else instruction.stem
        summary = next(
            (
                line.lstrip("#").strip()
                for line in text.splitlines()
                if line.lstrip().startswith("#")
            ),
            "",
        )
        return Skill(
            id=str(instruction.resolve()),
            name=name,
            path=str(instruction.resolve()),
            summary=summary,
            instructions=text,
            references=tuple(self._children(base / "references")),
            scripts=tuple(self._children(base / "scripts")),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    @staticmethod
    def _children(directory: Path) -> list[str]:
        try:
            return [
                str(path.resolve())
                for path in sorted(directory.iterdir(), key=lambda item: str(item))
                if path.is_file() and not path.is_symlink()
            ]
        except OSError:
            return []


__all__ = ["Skill", "SkillStore"]
