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


def print_thinking(text: str) -> None:
    print(style("\n  ~ thinking", DIM, CYAN))
    for line in text.splitlines():
        print(style(f"  | {line}", DIM))


def print_error(message: str) -> None:
    print(style(f"  ! {message}", RED))


def print_notice(message: str) -> None:
    print(style(f"  ~ {message}", DIM, CYAN))


class StreamPrinter:
    """Renders a streamed turn incrementally.

    Owns all the print-state bookkeeping: printing the reply label once,
    switching between thinking and answer text, and breaking the line for
    tool markers that arrive mid-stream. feed_* methods accept raw chunk
    content; call finish() when the turn's stream is exhausted.
    """

    def __init__(self, persona: str = "Caranthir", show_thinking: bool = False):
        self.persona = persona
        self.show_thinking = show_thinking
        self.mode: str | None = None  # None | "thinking" | "text"
        self._seen_hosted_ids: set[str] = set()

    # -- internal mode transitions ------------------------------------

    def _leave_mode(self) -> None:
        if self.mode == "thinking" and COLOR_ENABLED:
            sys.stdout.write(RESET)
        if self.mode is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self.mode = None

    def _enter(self, mode: str) -> None:
        if self.mode == mode:
            return
        self._leave_mode()
        if mode == "thinking":
            print(style("\n  ~ thinking", DIM, CYAN))
            if COLOR_ENABLED:
                sys.stdout.write(DIM)
        elif mode == "text":
            sys.stdout.write("\n" + style(self.persona, BOLD, MAGENTA) + style(" > ", DIM))
        sys.stdout.flush()
        self.mode = mode

    # -- feed methods ---------------------------------------------------

    def feed_ai_content(self, content: str | list) -> None:
        if isinstance(content, str):
            if content:
                self._enter("text")
                sys.stdout.write(content)
                sys.stdout.flush()
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "thinking":
                delta = block.get("thinking", "")
                if delta and self.show_thinking:
                    self._enter("thinking")
                    sys.stdout.write(delta)
                    sys.stdout.flush()
            elif btype == "text":
                delta = block.get("text", "")
                if delta:
                    self._enter("text")
                    sys.stdout.write(delta)
                    sys.stdout.flush()
            elif btype == "server_tool_use":
                block_id = block.get("id")
                name = block.get("name")
                if name and block_id and block_id not in self._seen_hosted_ids:
                    self._seen_hosted_ids.add(block_id)
                    self._leave_mode()
                    print_tool_call(f"{name} (hosted)", "running on provider servers")

    def feed_local_tool(self, name: str, result: str) -> None:
        self._leave_mode()
        print_tool_call(name, result)

    def finish(self) -> None:
        self._leave_mode()
        print()
