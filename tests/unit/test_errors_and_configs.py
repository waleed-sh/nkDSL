# Copyright (c) 2026 The neuraLQX and nkDSL Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import logging

import pytest

import nkdsl
from nkdsl.configs import UnknownOptionError
from nkdsl.errors import NKDSLError
from nkdsl.errors import SymbolicCompilerError
from nkdsl.errors import SymbolicOperatorExecutionError

pytestmark = pytest.mark.unit


def test_custom_error_messages_are_descriptive():
    err = NKDSLError("Primary", hint="Do X", details="Context Y")
    message = str(err)
    assert "Primary" in message
    assert "Hint: Do X" in message
    assert "Details: Context Y" in message

    exec_err = SymbolicOperatorExecutionError()
    assert "compile" in str(exec_err).lower()

    comp_err = SymbolicCompilerError()
    text = str(comp_err)
    assert "Symbolic compilation failed" in text
    assert "Check the symbolic IR" in text


def test_config_has_expected_debug_options_and_patch_semantics():
    opts = set(nkdsl.cfg.list_options())
    required = {
        "EXPERIMENTAL",
        "DEBUG",
        "DEBUG_VERBOSITY",
        "DEBUG_SCOPES",
        "DEBUG_PASSES",
        "DEBUG_LOG_TO_FILE",
        "DEBUG_LOG_DIR",
    }
    assert required.issubset(opts)

    old_debug = nkdsl.cfg.get("DEBUG")
    with nkdsl.cfg.patch(
        DEBUG=True,
        DEBUG_VERBOSITY="debug",
        DEBUG_SCOPES="compile,passes",
        DEBUG_PASSES="symbolic_validation",
        DEBUG_LOG_TO_FILE=False,
    ):
        assert nkdsl.cfg.get("DEBUG") is True
        assert nkdsl.cfg.get("DEBUG_VERBOSITY") == "debug"
        assert nkdsl.cfg.get("DEBUG_SCOPES") == ("compile", "passes")
        assert nkdsl.cfg.get("DEBUG_PASSES") == ("symbolic_validation",)
        assert nkdsl.cfg.get("DEBUG_LOG_TO_FILE") is False

    assert nkdsl.cfg.get("DEBUG") == old_debug


def test_config_unknown_option_and_thread_local_override():
    with pytest.raises(UnknownOptionError):
        nkdsl.cfg.get("DOES_NOT_EXIST")

    original = nkdsl.cfg.get("EXPERIMENTAL")
    with nkdsl.cfg.thread_local_override("EXPERIMENTAL", not original):
        assert nkdsl.cfg.get("EXPERIMENTAL") == (not original)
    assert nkdsl.cfg.get("EXPERIMENTAL") == original


def test_debug_settings_refresh_and_scope_filters():
    import nkdsl.debug as nkdebug

    with nkdsl.cfg.patch(
        DEBUG=True,
        DEBUG_SCOPES="compile,passes",
        DEBUG_PASSES="symbolic_validation",
        DEBUG_LOG_TO_FILE=False,
    ):
        nkdebug.refresh_settings(reinit=True)
        assert nkdebug.is_enabled()
        assert nkdebug.is_scope_enabled("compile")
        assert not nkdebug.is_scope_enabled("ir")
        assert nkdebug.is_scope_enabled("passes", pass_name="symbolic_validation")
        assert not nkdebug.is_scope_enabled("passes", pass_name="symbolic_normalization")

    nkdebug.refresh_settings(reinit=True)


def test_debug_false_disables_all_debug_emission(tmp_path):
    import nkdsl.debug as nkdebug

    before_len = len(nkdebug._EVENT_BUFFER)

    with nkdsl.cfg.patch(
        DEBUG=False,
        DEBUG_VERBOSITY="debug",
        DEBUG_SCOPES="all,passes",
        DEBUG_PASSES="symbolic_validation",
        DEBUG_LOG_TO_FILE=True,
        DEBUG_LOG_DIR=str(tmp_path),
    ):
        nkdebug.refresh_settings(reinit=True)
        assert nkdebug.is_enabled() is False
        assert nkdebug.get_logfile() is None

        nkdebug.event(
            "debug should be silent",
            scope="compile",
            level=logging.DEBUG,
            step="test",
        )

    assert len(nkdebug._EVENT_BUFFER) == before_len
    assert not any(tmp_path.iterdir())
