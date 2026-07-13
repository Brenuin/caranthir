"""ANSI terminal formatting for Caranthir. No dependencies beyond the stdlib."""

import sys
import time

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


def _drain_pending_console_input() -> str:
    """Return text already waiting in the console buffer right after Enter.

    Typing can't queue input that fast, so anything pending is the tail of a
    multi-line paste. Read it raw (echoing manually) rather than with input(),
    so a paste without a trailing newline can't leave us blocked waiting for
    an Enter that never comes.
    """
    try:
        import msvcrt
    except ImportError:
        return ""  # non-Windows console; paste stays line-by-line there
    chars: list[str] = []
    while True:
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # arrow/function key prefix: skip pair
                msvcrt.getwch()
                continue
            if ch == "\r":
                ch = "\n"
            chars.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
        # Large pastes can arrive in bursts; only stop once the buffer stays
        # empty across a short settle window.
        time.sleep(0.03)
        if not msvcrt.kbhit():
            break
    return "".join(chars).strip("\n")


def read_user_input() -> str:
    """Read one submission; a multi-line paste becomes a single prompt.

    With piped stdin (tests, scripts) this is exactly input(): one line per
    submission.
    """
    first = input(user_prompt_label())
    if not sys.stdin.isatty():
        return first
    rest = _drain_pending_console_input()
    return f"{first}\n{rest}" if rest else first


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
