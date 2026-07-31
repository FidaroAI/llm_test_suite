"""Questions that can be answered *or* backed out of.

Every prompt here returns its value or :data:`BACK`. That single convention is what lets the
wizard be one flat loop with an index cursor instead of nested calls that cannot be unwound
(see :mod:`porcelain.app`), so it is worth the key-binding work below.

Three behaviours are added on top of questionary:

* **Esc goes back.** Bound eagerly on each question's prompt_toolkit application. Safe with
  arrow keys: the vt100 parser resolves `\\x1b[A` into a `Up` key press before bindings are
  consulted, so a binding only ever sees a genuine lone Esc.
* **A prefilled default behaves like a highlighted one.** questionary puts a default into the
  edit buffer with the cursor after it, so typing *appends* — `60.0` plus `12.5` gives
  `60.012.5`. The first-keystroke binding clears the buffer for printable input, so Enter
  accepts the default and typing replaces it. Editing keys (backspace, arrows, Ctrl-U) are
  re-fed untouched, so they still edit rather than replace.
* **Ctrl-C is quitting, not backing out.** It propagates as `KeyboardInterrupt` and the app
  catches it once, at the top.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import questionary
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from questionary import Choice


class _Back:
    """The 'user pressed Esc' sentinel. A class so it reads well in a debugger."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "BACK"

    def __bool__(self) -> bool:
        # Falsy would invite `if answer:` checks that silently treat "go back" as "empty".
        # Truthy forces callers to compare against the sentinel explicitly.
        return True


BACK = _Back()

# questionary's default palette, plus a dim instruction line.
STYLE = questionary.Style([("instruction", "fg:#808080 italic")])

SELECT_HINT = "(arrows to move · Enter to choose · Esc to go back)"
CHECKBOX_HINT = "(space to toggle · Enter to confirm · Esc to go back)"
TEXT_HINT = "(Enter accepts the default · typing replaces it · Esc to go back)"


def _bind_escape(question: questionary.Question) -> None:
    """Make Esc exit `question` with :data:`BACK`.

    `merge_key_bindings` rather than `.add`, because select/checkbox expose a plain
    `KeyBindings` while text exposes a merged one with no `.add` method.
    """
    extra = KeyBindings()

    @extra.add("escape", eager=True)
    def _(event):
        event.app.exit(result=BACK)

    app = question.application
    app.key_bindings = merge_key_bindings([app.key_bindings, extra])


def _bind_replace_default(question: questionary.Question) -> None:
    """Make the first printable keystroke replace a prefilled default rather than append."""
    touched = {"yes": False}
    extra = KeyBindings()

    @extra.add("<any>", eager=True, filter=Condition(lambda: not touched["yes"]))
    def _(event):
        touched["yes"] = True
        data = event.data
        if data and data.isprintable():
            event.app.current_buffer.reset()
            event.app.current_buffer.insert_text(data)
        else:
            # Backspace, arrows, Ctrl-U, Ctrl-C: the user wants to edit or quit, not
            # replace. Re-feed so the binding that owns the key handles it as normal.
            event.key_processor.feed(event.key_sequence[0], first=True)

    app = question.application
    app.key_bindings = merge_key_bindings([app.key_bindings, extra])


def _ask(question: questionary.Question, *binders: Callable[[questionary.Question], None]):
    for binder in binders:
        binder(question)
    return question.unsafe_ask()


def select(message: str, choices: Sequence[Choice | str], default: Any = None) -> Any:
    """One choice from a list. Returns the chosen value, or :data:`BACK`."""
    question = questionary.select(
        message,
        choices=list(choices),
        default=default,
        style=STYLE,
        instruction=SELECT_HINT,
        use_shortcuts=False,
    )
    return _ask(question, _bind_escape)


def checkbox(message: str, choices: Sequence[Choice | str]) -> Any:
    """Zero or more choices. Re-asks on an empty selection; returns a list or :data:`BACK`.

    Re-asking rather than treating "nothing selected" as Esc: hitting Enter having selected
    nothing is nearly always "I forgot to press space", and silently backing out of the step
    would hide that instead of explaining it.
    """
    while True:
        question = questionary.checkbox(
            message, choices=list(choices), style=STYLE, instruction=CHECKBOX_HINT
        )
        answer = _ask(question, _bind_escape)
        if answer is BACK or answer:
            return answer
        print("  Nothing selected — use the space bar to pick at least one, or Esc to go back.")


def text(message: str, default: str = "") -> Any:
    """Free text with a pre-filled default. Enter accepts it; typing replaces it."""
    question = questionary.text(
        message, default=default, style=STYLE, instruction=TEXT_HINT if default else None
    )
    return _ask(question, _bind_escape, _bind_replace_default)
