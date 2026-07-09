"""ANSI terminal formatting for Caranthir. No dependencies beyond the stdlib."""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _supports_color() -> bool:
    return sys.stdout.isatty()


COLOR_ENABLED = _supports_color()

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"


def style(text: str, *codes: str) -> str:
    if not COLOR_ENABLED:
        return text
    return "".join(codes) + text + RESET


def print_banner(model: str, provider: str) -> None:
    title = "CARANTHIR"
    subtitle = f"{provider} / {model}"
    width = max(len(title), len(subtitle), 28) + 4
    rule = "─" * width
    print(style(rule, DIM, CYAN))
    print(style(f"  {title}", BOLD, CYAN))
    print(style(f"  {subtitle}", DIM))
    print(style(rule, DIM, CYAN))
    print(style("  Type /quit or /exit to leave.\n", DIM))


def user_prompt_label() -> str:
    return style("You", BOLD, GREEN) + style(" > ", DIM)


def print_tool_call(name: str, result: str) -> None:
    label = style("  * tool", DIM, YELLOW)
    print(f"{label} {style(name, YELLOW)} {style('->', DIM)} {result}")


def print_reply(persona: str, text: str) -> None:
    label = style(persona, BOLD, MAGENTA) + style(" > ", DIM)
    print(f"\n{label}{text}\n")


def print_error(message: str) -> None:
    print(style(f"  ! {message}", RED))
