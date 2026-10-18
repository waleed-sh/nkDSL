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

import jax
import numpy as np
import netket as nk
import pytest

import nkdsl
from nkdsl.compiler.diagnostics.collector import filter_diagnostics_by_minimum_severity
from nkdsl.compiler.diagnostics.formatting import format_diagnostic
from nkdsl.compiler.diagnostics.formatting import format_diagnostics_block
from nkdsl.compiler.diagnostics.formatting import linting_docs_url_for_code
from nkdsl.compiler.diagnostics.models import DSLDiagnostic
from nkdsl.compiler.diagnostics.models import count_diagnostics_by_severity
from nkdsl.compiler.diagnostics.models import diagnostic_severity_at_least
from nkdsl.compiler.diagnostics.models import normalize_diagnostic_severity
from nkdsl.compiler.diagnostics.state_sampling import build_hilbert_support_lookup
from nkdsl.compiler.diagnostics.state_sampling import evaluate_constraint_accepts_state
from nkdsl.compiler.diagnostics.state_sampling import illegal_local_state_positions
from nkdsl.compiler.diagnostics.state_sampling import sample_source_states
from nkdsl.compiler.diagnostics.state_sampling import state_to_tuple
import nkdsl.compiler.diagnostics.state_sampling as state_sampling_module
from nkdsl.compiler.diagnostics.rules.base import DiagnosticRuleContext
from nkdsl.compiler.diagnostics.rules.connectivity_rules import (
    GeneratedConnectivityValidityRule,
)
from nkdsl.compiler.core.options import SymbolicCompilerOptions

pytestmark = pytest.mark.unit


def test_diagnostic_models_and_formatting_helpers():
    diagnostic = DSLDiagnostic.create(
        code="NKDSL-X001",
        severity="warning",
        message="Something happened.",
        operator_name="op",
        term_name="t0",
        suggestion="Do something else.",
        context={"a": 1},
    )
    assert diagnostic.context_map()["a"] == 1
    assert normalize_diagnostic_severity("WARNING") == "warning"
    assert diagnostic_severity_at_least("error", "warning") is True
    formatted = format_diagnostic(diagnostic)
    assert "[NKDSL-X001]" in formatted
    assert "WARNING" in formatted
    assert "Suggestion:" in formatted
    assert (
        "Docs: https://nkdsl.readthedocs.io/en/latest/dsl/linting/messages.html#lint-code-nkdsl-x001"
        in formatted
    )
    formatted_block = format_diagnostics_block((diagnostic, diagnostic), max_items=1)
    assert "more diagnostic" in formatted_block
    assert (
        "Read more: https://nkdsl.readthedocs.io/en/latest/dsl/linting/messages.html"
        in formatted_block
    )
    assert (
        "Overview: https://nkdsl.readthedocs.io/en/latest/dsl/linting/index.html" in formatted_block
    )
    no_suggestion = DSLDiagnostic.create(
        code="NKDSL-X002",
        severity="info",
        message="No suggestion.",
        operator_name="op",
    )
    assert "Suggestion:" not in format_diagnostic(no_suggestion)
    assert (
        "Docs: https://nkdsl.readthedocs.io/en/latest/dsl/linting/messages.html#lint-code-nkdsl-x002"
        in format_diagnostic(no_suggestion)
    )
    assert format_diagnostics_block((), max_items=1) == "No DSL diagnostics."
    assert (
        linting_docs_url_for_code("NKDSL-E001")
        == "https://nkdsl.readthedocs.io/en/latest/dsl/linting/messages.html#lint-code-nkdsl-e001"
    )

    counts = count_diagnostics_by_severity((diagnostic,))
    assert counts["warning"] == 1
    filtered = filter_diagnostics_by_minimum_severity((diagnostic,), minimum="error")
    assert filtered == ()

    with pytest.raises(ValueError, match="Unsupported diagnostics severity"):
        normalize_diagnostic_severity("trace")
    with pytest.raises(ValueError, match="non-empty string"):
        DSLDiagnostic.create(code=" ", severity="warning", message="x", operator_name="op")
    with pytest.raises(ValueError, match="non-empty string"):
        DSLDiagnostic.create(code="X", severity="warning", message=" ", operator_name="op")
    with pytest.raises(ValueError, match="non-empty string"):
        DSLDiagnostic.create(code="X", severity="warning", message="x", operator_name=" ")
    with pytest.raises(ValueError, match="term_name must be non-empty"):
        DSLDiagnostic.create(
            code="X", severity="warning", message="x", operator_name="op", term_name=" "
        )
    with pytest.raises(ValueError, match="suggestion must be non-empty"):
        DSLDiagnostic.create(
            code="X", severity="warning", message="x", operator_name="op", suggestion=" "
        )
    assert "DSLDiagnostic(" in repr(diagnostic)


def test_state_sampling_helpers_and_support_lookup():
    hi = nk.hilbert.Fock(n_max=1, N=2, n_particles=1)
    samples = sample_source_states(hi, max_samples=4)
    assert samples is not None
    assert samples.shape[1] == 2

    support = build_hilbert_support_lookup(hi, max_states=64)
    assert support is not None
    assert tuple(samples[0].tolist()) in support
    assert build_hilbert_support_lookup(hi, max_states=1) is None

    constraint_ok = evaluate_constraint_accepts_state(hi, np.asarray([1, 0], dtype=np.int32))
    constraint_bad = evaluate_constraint_accepts_state(hi, np.asarray([1, 1], dtype=np.int32))
    assert constraint_ok is True
    assert constraint_bad is False

    illegal = illegal_local_state_positions(np.asarray([0, 3], dtype=np.int32), hi.local_states)
    assert illegal == (1,)
    assert state_to_tuple(np.asarray([1, 0], dtype=np.int32)) == (1, 0)


def test_state_sampling_internal_edge_cases():
    assert state_sampling_module._safe_int("bad") is None
    assert state_sampling_module._state_rows(np.asarray([1, 2])).shape == (1, 2)
    assert state_sampling_module._state_rows(np.zeros((1, 2, 3))) is None
    assert sample_source_states(nk.hilbert.Fock(n_max=1, N=1), max_samples=0) is None

    class _NumbersOnlyHilbert:
        n_states = 5

        def numbers_to_states(self, numbers):
            numbers = np.asarray(numbers, dtype=np.int64)
            return np.stack([numbers, numbers + 1], axis=1)

    rows = sample_source_states(_NumbersOnlyHilbert(), max_samples=3)
    assert rows is not None
    assert rows.shape == (3, 2)

    class _BrokenNumbersHilbert:
        n_states = 3

        def numbers_to_states(self, numbers):
            raise RuntimeError("broken")

    assert sample_source_states(_BrokenNumbersHilbert(), max_samples=3) is None

    class _BrokenAllStatesHilbert:
        n_states = 2

        def all_states(self):
            raise RuntimeError("broken")

    assert build_hilbert_support_lookup(_BrokenAllStatesHilbert(), max_states=4) is None

    class _ThreeDimAllStatesHilbert:
        n_states = 2

        def all_states(self):
            return np.zeros((1, 1, 1), dtype=np.int32)

    assert build_hilbert_support_lookup(_ThreeDimAllStatesHilbert(), max_states=4) is None

    class _NoConstraint:
        constraint = None

    assert (
        evaluate_constraint_accepts_state(_NoConstraint(), np.asarray([0], dtype=np.int32)) is None
    )

    class _BrokenConstraint:
        def constraint(self, x):
            raise RuntimeError("bad")

    assert (
        evaluate_constraint_accepts_state(_BrokenConstraint(), np.asarray([0], dtype=np.int32))
        is None
    )

    class _EmptyConstraint:
        def constraint(self, x):
            return np.asarray([], dtype=bool)

    assert (
        evaluate_constraint_accepts_state(_EmptyConstraint(), np.asarray([0], dtype=np.int32))
        is None
    )
    assert illegal_local_state_positions(np.asarray([1], dtype=np.int32), None) == ()
    assert illegal_local_state_positions(np.asarray([1], dtype=np.int32), np.zeros((1, 1))) == ()


def test_non_indexable_hilbert_diagnostics_do_not_access_indexing_apis():
    class _NonIndexableHilbert:
        is_indexable = False

        @property
        def n_states(self):
            pytest.fail("Diagnostics must not read non-indexable cardinality")

        def all_states(self):
            pytest.fail("Diagnostics must not enumerate a non-indexable space")

        def numbers_to_states(self, numbers):
            pytest.fail("Diagnostics must not index a non-indexable space")

    hi = _NonIndexableHilbert()
    assert sample_source_states(hi, max_samples=4) is None
    assert build_hilbert_support_lookup(hi, max_states=64) is None


def test_non_indexable_hilbert_uses_reproducible_direct_sampling():
    class _SampleableHilbert:
        is_indexable = False

        @property
        def n_states(self):
            pytest.fail("Direct sampling must not read Hilbert cardinality")

        def random_state(self, key, size):
            assert size == 4
            return jax.random.randint(key, (size, 8), 0, 2)

    hi = _SampleableHilbert()
    rows = sample_source_states(hi, max_samples=4)
    assert rows.shape == (4, 8)
    np.testing.assert_array_equal(rows, sample_source_states(hi, max_samples=4))
    assert build_hilbert_support_lookup(hi, max_states=64) is None


def test_large_constrained_hilbert_uses_netket_direct_sampling():
    hi = nk.hilbert.Fock(n_max=1, N=64, n_particles=32)
    assert not hi.is_indexable
    rows = sample_source_states(hi, max_samples=4)
    assert rows.shape == (4, 64)
    np.testing.assert_array_equal(np.sum(rows, axis=1), 32)


def test_generic_custom_constraint_rejection_sampling_is_not_invoked():
    class _UnavailableConstraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
        def __call__(self, states):
            pytest.fail("Diagnostics must not enter NetKet's unbounded rejection sampler")

    hi = nk.hilbert.HomogeneousHilbert(
        nk.utils.StaticRange(start=0, step=1, length=2),
        N=64,
        constraint=_UnavailableConstraint(),
    )
    assert not hi.is_indexable
    assert sample_source_states(hi, max_samples=4) is None


@pytest.mark.parametrize("attribute", ("is_indexable", "n_states"))
def test_diagnostics_skip_unavailable_indexing_metadata(attribute):
    def unavailable(self):
        raise RuntimeError("The hilbert space is too large to be indexed.")

    hilbert_type = type("UnavailableIndex", (), {attribute: property(unavailable)})
    hi = hilbert_type()
    assert sample_source_states(hi, max_samples=4) is None
    assert build_hilbert_support_lookup(hi, max_states=64) is None


def test_source_state_sampling_requests_only_a_bounded_index_prefix():
    class _LargeIndexableHilbert:
        is_indexable = True
        n_states = 2**30

        def all_states(self):
            pytest.fail("A small diagnostic sample must not enumerate the full space")

        def numbers_to_states(self, numbers):
            np.testing.assert_array_equal(numbers, np.arange(4))
            return np.stack((numbers, numbers + 1), axis=1)

    rows = sample_source_states(_LargeIndexableHilbert(), max_samples=4)
    np.testing.assert_array_equal(rows, np.stack((np.arange(4), np.arange(1, 5)), axis=1))


def test_all_states_sampling_fallback_requires_a_small_known_cardinality():
    class _SmallHilbert:
        n_states = 2

        def all_states(self):
            return np.asarray([[0, 1], [1, 0]])

    np.testing.assert_array_equal(
        sample_source_states(_SmallHilbert(), max_samples=4), [[0, 1], [1, 0]]
    )
    assert sample_source_states(_SmallHilbert(), max_samples=1) is None


def test_failed_direct_sampling_falls_back_to_bounded_indexing():
    class _BrokenSamplerHilbert:
        is_indexable = True
        n_states = 2

        def random_state(self, key, size):
            raise NotImplementedError("Sampling is not implemented")

        def numbers_to_states(self, numbers):
            return np.asarray(numbers)[:, None]

    np.testing.assert_array_equal(
        sample_source_states(_BrokenSamplerHilbert(), max_samples=4), [[0], [1]]
    )


def test_connectivity_rule_reports_unavailable_source_sampling():
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(nk.hilbert.Fock(n_max=1, N=1), "no-samples")
        .for_each_site("i")
        .emit(matrix_element=1.0)
        .build()
    )
    context = DiagnosticRuleContext(
        operator=type("Op", (), {"hilbert": object()})(),
        ir=op.to_ir(),
        options=SymbolicCompilerOptions(),
    )
    diagnostics = GeneratedConnectivityValidityRule().run(context)
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "NKDSL-I303"
    assert diagnostics[0].severity == "info"


def test_traversal_helpers_cover_static_index_nodes():
    term = nkdsl.SymbolicIRTerm.create(
        name="diag",
        iterator=nkdsl.KBodyIteratorSpec(labels=(), index_sets=((),)),
        predicate=nkdsl.PredicateExpr.constant(True),
        update_program=nkdsl.identity().to_program(),
        amplitude=nkdsl.source_index(1) + nkdsl.target_index(0),
    )
    from nkdsl.compiler.diagnostics.traversals import iter_term_static_index_nodes

    nodes = list(iter_term_static_index_nodes(term))
    assert ("static_index", 1) in nodes
    assert ("static_emitted_index", 0) in nodes


def test_connectivity_rule_internal_edge_paths():
    hi = nk.hilbert.Fock(n_max=1, N=1)
    term = nkdsl.SymbolicIRTerm.create(
        name="diag",
        iterator=nkdsl.KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=nkdsl.PredicateExpr.gt(nkdsl.symbol("missing"), 0),
        update_program=nkdsl.identity().to_program(),
        amplitude=1.0,
    )
    ir = nkdsl.SymbolicOperatorIR.from_terms(
        operator_name="edge-connectivity",
        hilbert_size=1,
        dtype_str="float64",
        is_hermitian=False,
        terms=(term,),
    )
    context = DiagnosticRuleContext(
        operator=type("Op", (), {"hilbert": hi})(),
        ir=ir,
        options=SymbolicCompilerOptions(
            strict_validation=False,
            cache_enabled=False,
            lint_state_sample_size=4,
            lint_branch_sample_cap=1,
            lint_max_exact_hilbert_states=1,
            diagnostics_min_severity="info",
        ),
    )
    diagnostics = GeneratedConnectivityValidityRule().run(context)
    codes = {item.code for item in diagnostics}
    assert "NKDSL-I301" in codes
    assert "NKDSL-I302" in codes

    term_update_error = nkdsl.SymbolicIRTerm.create(
        name="update_error",
        iterator=nkdsl.KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=nkdsl.PredicateExpr.constant(True),
        update_program=nkdsl.write("i", nkdsl.symbol("delta")).to_program(),
        amplitude=1.0,
    )
    ir_update = nkdsl.SymbolicOperatorIR.from_terms(
        operator_name="update-error",
        hilbert_size=1,
        dtype_str="float64",
        is_hermitian=False,
        terms=(term_update_error,),
    )
    context_update = DiagnosticRuleContext(
        operator=type("Op", (), {"hilbert": hi})(),
        ir=ir_update,
        options=SymbolicCompilerOptions(
            strict_validation=False,
            cache_enabled=False,
            lint_state_sample_size=4,
            lint_branch_sample_cap=2,
            lint_max_exact_hilbert_states=16,
            diagnostics_min_severity="info",
        ),
    )
    diagnostics_update = GeneratedConnectivityValidityRule().run(context_update)
    assert any(item.code == "NKDSL-I302" for item in diagnostics_update)

    cap_term = nkdsl.SymbolicIRTerm.create(
        name="cap",
        iterator=nkdsl.KBodyIteratorSpec(labels=("i",), index_sets=((0,),)),
        predicate=nkdsl.PredicateExpr.constant(True),
        update_program=nkdsl.shift("i", +2).to_program(),
        amplitude=1.0,
    )
    ir_cap = nkdsl.SymbolicOperatorIR.from_terms(
        operator_name="cap",
        hilbert_size=1,
        dtype_str="float64",
        is_hermitian=False,
        terms=(cap_term,),
    )
    context_cap = DiagnosticRuleContext(
        operator=type("Op", (), {"hilbert": hi})(),
        ir=ir_cap,
        options=SymbolicCompilerOptions(
            strict_validation=False,
            cache_enabled=False,
            lint_state_sample_size=8,
            lint_branch_sample_cap=1,
            lint_max_exact_hilbert_states=16,
            diagnostics_min_severity="warning",
        ),
    )
    diagnostics_cap = GeneratedConnectivityValidityRule().run(context_cap)
    assert any(item.code == "NKDSL-W303" for item in diagnostics_cap)
