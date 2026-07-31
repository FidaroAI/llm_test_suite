"""The questionary layer, driven through a prompt_toolkit pipe — no real terminal needed.

Worth testing despite being a thin wrapper: the Esc binding is what the whole navigation
model rests on, and the "typing replaces the default" binding is easy to break by accident.
Both are custom key bindings against a third-party library, so they are exactly the kind of
thing that stops working on an upgrade without anyone noticing.
"""

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from questionary import Choice

from llmevalx import prompts
from llmevalx.prompts import BACK

ESC = "\x1b"
CR = "\r"
SPACE = " "
UP = "\x1b[A"
DOWN = "\x1b[B"
BACKSPACE = "\x7f"
CTRL_C = "\x03"
CTRL_U = "\x15"


def drive(ask, keys):
    """Run `ask()` with `keys` fed in as if typed."""
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        with create_app_session(input=pipe, output=DummyOutput()):
            return ask()


def a_select():
    return prompts.select("pick", [Choice("A", "a"), Choice("B", "b"), Choice("C", "c")])


def a_checkbox():
    return prompts.checkbox("pick", [Choice("A", "a"), Choice("B", "b")])


def a_text(default="60.0"):
    return lambda: prompts.text("value", default=default)


# --------------------------------------------------------------------------- Esc


def test_esc_backs_out_of_a_select():
    assert drive(a_select, ESC) is BACK


def test_esc_backs_out_of_a_checkbox():
    assert drive(a_checkbox, ESC) is BACK


def test_esc_backs_out_of_a_text_prompt():
    assert drive(a_text(), ESC) is BACK


def test_esc_after_moving_still_backs_out():
    assert drive(a_select, DOWN + DOWN + ESC) is BACK


# --------------------------------------------------------------------------- arrow keys


def test_arrow_keys_still_move_the_cursor():
    """The eager Esc binding must not swallow arrow keys, which are escape sequences."""
    assert drive(a_select, DOWN + DOWN + CR) == "c"


def test_arrows_move_both_ways():
    assert drive(a_select, DOWN + DOWN + UP + CR) == "b"


def test_enter_takes_the_first_choice_by_default():
    assert drive(a_select, CR) == "a"


# --------------------------------------------------------------------------- checkbox


def test_space_toggles_and_enter_confirms():
    assert drive(a_checkbox, SPACE + DOWN + SPACE + CR) == ["a", "b"]


def test_checkbox_reasks_when_nothing_is_selected(capsys):
    """Enter on an empty list is a slip, not a request to go back."""
    result = drive(a_checkbox, CR + SPACE + CR)
    assert result == ["a"]
    assert "Nothing selected" in capsys.readouterr().out


# Not tested here: pressing Esc on the *re-asked* checkbox. The re-ask builds a second
# prompt_toolkit application, and keys queued on the pipe before the first one exits do not
# survive the hand-over — the test hangs waiting for input that was already consumed. It is a
# limitation of driving two applications from one pre-loaded pipe, not of the prompt: the
# re-ask goes through the same `_ask` path that `test_esc_backs_out_of_a_checkbox` covers.


# --------------------------------------------------------------------------- defaults


def test_enter_accepts_the_prefilled_default():
    assert drive(a_text(), CR) == "60.0"


def test_typing_replaces_the_default_rather_than_appending():
    assert drive(a_text(), "12.5" + CR) == "12.5"


def test_backspace_edits_the_default_instead_of_replacing_it():
    """Backspace says 'I want to change this one', not 'start over'."""
    assert drive(a_text(), BACKSPACE + "7" + CR) == "60.7"


def test_ctrl_u_clears_the_default():
    assert drive(a_text(), CTRL_U + "9" + CR) == "9"


def test_editing_continues_normally_after_the_first_keystroke():
    assert drive(a_text(), "125" + BACKSPACE + CR) == "12"


def test_an_empty_default_behaves_like_an_ordinary_prompt():
    assert drive(a_text(default=""), "abc" + CR) == "abc"


# --------------------------------------------------------------------------- quitting


def test_ctrl_c_quits_rather_than_backing_out():
    with pytest.raises(KeyboardInterrupt):
        drive(a_text(), CTRL_C)


def test_ctrl_c_in_a_select_quits():
    with pytest.raises(KeyboardInterrupt):
        drive(a_select, CTRL_C)


# --------------------------------------------------------------------------- the sentinel


def test_back_is_truthy_so_it_cannot_be_mistaken_for_an_empty_answer():
    assert bool(BACK) is True


def test_back_is_a_singleton():
    assert prompts._Back() is BACK
