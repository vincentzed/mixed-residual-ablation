#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["pydantic>=2.9", "rich>=13.7", "typer>=0.12"]
# ///
"""Single source of truth for the humans& palette.

`assets/tokens.css` is the only place a hex is written down. Everything else --
the figure scripts, the reference docs, any artifact CSS -- derives from it.

Before this existed the palette lived in three unlinked copies (tokens.css,
references/visual-system.md, scripts/thread_visuals.py). They agreed, which is
the dangerous state: nothing forced them to keep agreeing, and nothing would
have announced it when they stopped.

    palette.py show                 # both themes, as a table
    palette.py emit --format python # paste-ready constants
    palette.py check FILE...        # non-zero if a file's hexes contradict tokens.css
    palette.py check --all          # check every known consumer

`check` is the point. Wire it into CI or run it after editing tokens.css; it
reports every hex that a consumer states differently from the source, so drift
is caught the day it happens instead of the day someone notices a figure looks
slightly wrong.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Annotated, Literal

from rich.console import Console
from rich.table import Table
from rich.text import Text

import typer
from pydantic import BaseModel, Field

SKILL_ROOT = Path(__file__).resolve().parent.parent
TOKENS_CSS = SKILL_ROOT / "assets" / "tokens.css"

# Consumers are DISCOVERED, not listed. A hardcoded list is an authority that never
# looks at the territory: a new file hardcoding a colour is simply never checked, and
# `check` reports "No drift" over a directory containing drift. Same defect as a
# hardcoded root list -- the verdict does not carry its bound.
#
# tokens.css is the source, and this file holds EXEMPT, so both are excluded.
_NOT_CONSUMERS = frozenset({"assets/tokens.css", "scripts/palette.py"})
# Skip what is definitely not a consumer, rather than enumerating what is. An earlier
# version listed suffixes (.py/.md/.css/...) -- complete for the tree that existed when
# it was written, and silently blind to the next one. That is the same allowlist defect
# this project already hit once: naming what you exclude fails safe, naming what you
# include fails silent. Binary files are excluded by failing to decode, not by name.
_SKIP_DIRS = frozenset({".ruff_cache", "__pycache__", ".git", "node_modules"})
_MAX_BYTES = 2_000_000


def discover_consumers() -> list[Path]:
    """Every readable file in the skill that states a colour, bar the source and this one."""
    out: list[Path] = []
    for path in sorted(SKILL_ROOT.rglob("*")):
        if not path.is_file() or _SKIP_DIRS & set(path.parts):
            continue
        rel = path.relative_to(SKILL_ROOT).as_posix()
        if rel in _NOT_CONSUMERS:
            continue
        try:
            if path.stat().st_size > _MAX_BYTES:
                continue
            if _HEX.search(path.read_text(encoding="utf-8")):
                out.append(path)
        except (OSError, UnicodeDecodeError):
            continue  # unreadable or binary -- cannot state a colour in text form
    return out


Theme = Literal["light", "dark"]

# tokens.css declares light under `:root` / `[data-theme="light"]` and dark under
# `@media (prefers-color-scheme: dark)` / `[data-theme="dark"]`.
_DARK_CONTEXT = re.compile(
    r'prefers-color-scheme:\s*dark|\[data-theme=["\']dark["\']\]'
)
_DECL = re.compile(r"(--[a-z0-9-]+)\s*:\s*([^;]+);")
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")

# Hexes that appear in consumers for reasons unrelated to the palette. Each needs
# a stated reason -- an unexplained entry here is how a real drift gets silenced.
EXEMPT: dict[str, str] = {
    "#0d0d10": "documented rejected color (old 'unified black')",
    "#1e1a24": "documented rejected color (old purple card)",
    "#a78bfa": "documented rejected color (old purple hero)",
    "#292929": "categorical_grid surface -- documented 8-slot exception",
    "#3987e5": "categorical_grid slot -- documented 8-slot exception",
    "#d95926": "categorical_grid slot -- documented 8-slot exception",
    "#199e70": "categorical_grid slot -- documented 8-slot exception",
    "#c98500": "categorical_grid slot -- documented 8-slot exception",
    "#d55181": "categorical_grid slot -- documented 8-slot exception",
    "#008300": "categorical_grid slot -- documented 8-slot exception",
    "#9085e9": "categorical_grid slot -- documented 8-slot exception",
    "#e66767": "categorical_grid slot -- documented 8-slot exception",
}


class Palette(BaseModel):
    """The token set for one theme, parsed from tokens.css."""

    theme: Theme
    tokens: dict[str, str] = Field(default_factory=dict)

    @property
    def hexes(self) -> set[str]:
        return {v.lower() for v in self.tokens.values() if _HEX.fullmatch(v)}

    def constant_name(self, token: str) -> str:
        return token.removeprefix("--").replace("-", "_").upper()


class Drift(BaseModel):
    """One hex a consumer states that tokens.css does not."""

    file: Path
    line: int
    hex: str
    text: str


def parse_tokens(css: Path) -> dict[Theme, Palette]:
    """Split tokens.css into light and dark token sets.

    Values are read per declaration block; a block counts as dark when its
    opening context matches a dark selector or media query.
    """
    source = css.read_text(encoding="utf-8")
    out: dict[Theme, Palette] = {
        "light": Palette(theme="light"),
        "dark": Palette(theme="dark"),
    }

    depth = 0
    dark_depth: int | None = None
    for raw in source.splitlines():
        line = raw.split("/*")[0]
        opening = _DARK_CONTEXT.search(raw) is not None

        for token, value in _DECL.findall(line):
            theme: Theme = "dark" if dark_depth is not None else "light"
            out[theme].tokens[token] = value.strip()

        depth += line.count("{")
        if opening and dark_depth is None:
            dark_depth = depth
        depth -= line.count("}")
        if dark_depth is not None and depth < dark_depth:
            dark_depth = None

    # Every dark token inherits its light counterpart unless overridden.
    for token, value in out["light"].tokens.items():
        out["dark"].tokens.setdefault(token, value)
    return out


def find_drift(path: Path, allowed: set[str]) -> list[Drift]:
    """Every hex in `path` that tokens.css does not contain and is not exempt."""
    drifts: list[Drift] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for found in _HEX.findall(line):
            value = found.lower()
            if value in allowed or value in EXEMPT:
                continue
            drifts.append(
                Drift(file=path, line=number, hex=value, text=line.strip()[:90])
            )
    return drifts


app = typer.Typer(add_completion=False, help=__doc__)
console = Console()
err = Console(stderr=True)


@app.command()
def show() -> None:
    """Print both themes side by side."""
    palettes = parse_tokens(TOKENS_CSS)
    table = Table(title="humans& tokens — assets/tokens.css", header_style="bold")
    table.add_column("token")
    table.add_column("light")
    table.add_column("dark")

    for token in palettes["light"].tokens:
        light = palettes["light"].tokens[token]
        dark = palettes["dark"].tokens.get(token, "")
        table.add_row(
            token,
            Text(light, style=light if _HEX.fullmatch(light) else ""),
            Text(dark, style=dark if _HEX.fullmatch(dark) else ""),
        )
    console.print(table)


@app.command()
def emit(
    fmt: Annotated[
        str, typer.Option("--format", help="python | json | css")
    ] = "python",
    theme: Annotated[str, typer.Option(help="light | dark | both")] = "both",
) -> None:
    """Write the palette out in a form another file can consume."""
    palettes = parse_tokens(TOKENS_CSS)
    wanted: list[Theme] = ["light", "dark"] if theme == "both" else [theme]  # type: ignore[list-item]

    match fmt:
        case "json":
            import json

            payload = {t: palettes[t].tokens for t in wanted}
            console.print_json(json.dumps(payload))
        case "css":
            for name in wanted:
                console.print(f":root[data-theme='{name}'] {{")
                for token, value in palettes[name].tokens.items():
                    console.print(f"  {token}: {value};")
                console.print("}")
        case _:
            console.print("# Generated by scripts/palette.py — do not hand-edit.")
            console.print("# Source of truth: assets/tokens.css")
            for name in wanted:
                console.print(f"\n{name.upper()} = {{")
                for token, value in palettes[name].tokens.items():
                    console.print(
                        f'    "{palettes[name].constant_name(token)}": "{value}",'
                    )
                console.print("}")


@app.command()
def check(
    files: Annotated[list[Path] | None, typer.Argument(help="Files to check")] = None,
    every: Annotated[
        bool, typer.Option("--all", help="Check every known consumer")
    ] = False,
) -> None:
    """Fail if any file states a hex that tokens.css does not.

    This is the guard that makes the three-copy problem structurally impossible.
    """
    palettes = parse_tokens(TOKENS_CSS)
    allowed = palettes["light"].hexes | palettes["dark"].hexes

    # A check that cannot fail is not a check. With an empty source every hex is
    # "known" vacuously and every consumer passes -- so a truncated tokens.css
    # would report No drift. Nothing disagrees with nothing.
    if not allowed:
        err.print(
            f"[red]{TOKENS_CSS} yielded no colour tokens.[/red] The source of truth is "
            "empty or unparseable, so 'no drift' would be vacuous. Refusing to report."
        )
        raise typer.Exit(2)

    targets = discover_consumers() if every or not files else list(files)
    missing = [t for t in targets if not t.is_file()]
    if missing:
        for path in missing:
            err.print(f"[red]missing consumer:[/red] {path}")
        raise typer.Exit(2)

    drifts = [d for target in targets for d in find_drift(target, allowed)]

    # A consumer stating no hexes at all agrees with everything vacuously, which is
    # exactly what an emptied or truncated file looks like. Say so, don't print "ok".
    empty: list[Path] = []
    for target in targets:
        count = sum(1 for d in drifts if d.file == target)
        stated = len(_HEX.findall(target.read_text(encoding="utf-8")))
        if not stated:
            empty.append(target)
        mark = (
            "[red]DRIFT[/red]"
            if count
            else "[yellow]EMPTY[/yellow]"
            if not stated
            else "[green]ok[/green]"
        )
        console.print(
            f"{mark}  {target.relative_to(SKILL_ROOT)}  "
            f"({count} unknown of {stated} stated)"
        )

    if drifts:
        table = Table(title="hexes not present in tokens.css", header_style="bold red")
        for column in ("file", "line", "hex", "context"):
            table.add_column(column)
        for d in drifts:
            table.add_row(
                str(d.file.relative_to(SKILL_ROOT)), str(d.line), d.hex, d.text
            )
        console.print(table)
        err.print(
            "\nEither the consumer is wrong, or tokens.css changed and the consumer "
            "was not regenerated. Fix tokens.css first — it is the source."
        )
        raise typer.Exit(1)

    if empty:
        err.print(
            f"\n[yellow]{len(empty)} consumer(s) state no colours at all.[/yellow] "
            "They agree vacuously; that is indistinguishable from a file that was "
            "emptied. Treating as a failure."
        )
        raise typer.Exit(1)

    console.print(
        "[dim]consumers discovered: "
        + ", ".join(str(t.relative_to(SKILL_ROOT)) for t in targets)
        + "[/dim]"
    )
    console.print(
        f"\n[green]No drift.[/green] {len(targets)} consumer(s) agree with "
        f"{TOKENS_CSS.relative_to(SKILL_ROOT)} ({len(allowed)} tokens)."
    )


if __name__ == "__main__":
    if not TOKENS_CSS.is_file():
        err.print(f"[red]tokens.css not found at {TOKENS_CSS}[/red]")
        sys.exit(2)
    app()
