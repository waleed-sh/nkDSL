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

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

import nkdsl
from nkdsl.compiler.core.context import SymbolicCompilationContext
from nkdsl.compiler.lowering.coalescing import (
    coalesce_connected_components,
    compact_connections,
    pack_integer_keys,
)
from nkdsl.compiler.passes.connectivity_cost import SymbolicConnectivityCostPass
from nkdsl.compiler.passes.deduplication import SymbolicDeduplicationPass, analyze_connectivity
from nkdsl.compiler.passes.analysis import SymbolicMaxConnSizeAnalysisPass
from nkdsl.errors import SymbolicCompilerError, SymbolicDiagnosticsError

pytestmark = pytest.mark.unit


def _compile(symbolic, dedup=True, **options):
    return nkdsl.SymbolicCompiler(
        options=nkdsl.SymbolicCompilerOptions(
            cache_enabled=False,
            diagnostics_enabled=False,
            deduplicate_connected_components=dedup,
            **options,
        )
    ).compile(symbolic)


def _assert_equivalent(symbolic, inputs=None):
    if inputs is None:
        inputs = jnp.asarray(symbolic.hilbert.all_states())
    raw = _compile(symbolic, False).compiled_operator
    artifact = _compile(symbolic)
    compiled = artifact.compiled_operator
    xp_raw, m_raw = map(np.asarray, jax.block_until_ready(jax.jit(raw.get_conn_padded)(inputs)))
    xp, mels = map(np.asarray, jax.block_until_ready(jax.jit(compiled.get_conn_padded)(inputs)))
    assert xp.shape[-2] == mels.shape[-1] == compiled.max_conn_size
    for original, amplitudes, actual, weights in zip(xp_raw, m_raw, xp, mels):
        expected = {}
        for state, amplitude in zip(original, amplitudes):
            key = tuple(state)
            expected[key] = expected.get(key, 0) + amplitude
        result = {}
        for state, amplitude in zip(actual, weights):
            if amplitude != 0:
                key = tuple(state)
                assert key not in result
                result[key] = amplitude
        for key in expected.keys() | result.keys():
            np.testing.assert_allclose(
                expected.get(key, 0), result.get(key, 0), atol=1e-10, rtol=1e-6
            )
        active = np.flatnonzero(weights != 0)
        np.testing.assert_array_equal(active, np.arange(len(active)))
        assert np.all(actual[len(active) :] == 0)
    return artifact


def test_diagonal_classes_reduce_output_bound_and_remove_runtime_sorts():
    hi = nk.hilbert.Fock(n_max=3, N=5)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "diagonal")
        .for_each_site("i")
        .emit(matrix_element=nkdsl.site("i").value)
        .build()
    )
    artifact = _assert_equivalent(symbolic)
    assert artifact.compiled_operator.max_conn_size == 1
    plan = artifact.metadata_map()["connectivity_deduplication"]
    assert plan["raw_size"] == 5
    assert plan["runtime_component_sizes"] == ()
    hlo = jax.jit(artifact.compiled_operator.get_conn_padded).lower(
        jnp.ones((3, 5), dtype=jnp.int8)
    )
    assert " sort(" not in hlo.compiler_ir(dialect="hlo").as_hlo_text()


@pytest.mark.parametrize("kind", ["hop", "flip", "modular", "guarded_swap", "swap", "copies"])
def test_operator_families_match_raw_for_entire_hilbert_space(kind):
    hi = nk.hilbert.Fock(n_max=3, N=3)
    a, b = nkdsl.site("i").value, nkdsl.site("j").value
    builder = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, kind)
        .for_each_site("k")
        .emit(
            matrix_element=nkdsl.site("k").value - 1,
        )
    )
    if kind == "hop":
        builder = (
            builder.for_each(("i", "j"), over=[(0, 1), (1, 2)])
            .where((a > 0) & (b < 3))
            .emit(nkdsl.hop("i", "j"), matrix_element=(a * (b + 1)).sqrt())
        )
    elif kind == "flip":
        builder = builder.for_each_site("i").emit(
            nkdsl.affine("i", scale=-1, bias=3), matrix_element=1.0
        )
    elif kind == "modular":
        builder = builder.for_each_site("i").emit(nkdsl.shift_mod("i", 1), matrix_element=1.0)
    elif kind in {"guarded_swap", "swap"}:
        builder = builder.for_each(("i", "j"), over=[(0, 1), (1, 2)])
        if kind == "guarded_swap":
            builder = builder.where(a != b)
        builder = builder.emit(nkdsl.swap("i", "j"), matrix_element=1.0)
    else:
        builder = builder.for_each_site("i").emit(nkdsl.write(0, a), matrix_element=1.0)
    _assert_equivalent(builder.build())


def test_finite_copy_images_reduce_dynamic_output_capacity():
    hi = nk.hilbert.Fock(n_max=1, N=8)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "copy-image")
        .for_each_site("i")
        .emit(nkdsl.write(0, nkdsl.site("i").value), matrix_element=1.0)
        .build()
    )
    artifact = _assert_equivalent(symbolic)
    assert artifact.compiled_operator.max_conn_size == 2
    assert artifact.metadata_map()["connectivity_deduplication"][
        "runtime_component_capacities"
    ] == (2,)
    assert artifact.metadata_map()["connectivity_cost"]["runtime_strategies"] == ("workspace",)


def test_equivalent_programs_across_labels_and_sequential_permutations():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "value-numbering")
        .for_each_site("i")
        .emit(nkdsl.shift("i", 1), matrix_element=1.0)
        .for_each_site("j")
        .emit(nkdsl.shift("j", 1), matrix_element=2.0)
        .globally()
        .emit(nkdsl.swap(0, 1).swap(0, 1), matrix_element=3.0)
        .emit(nkdsl.identity(), matrix_element=4.0)
        .build()
    )
    artifact = _assert_equivalent(symbolic)
    assert artifact.compiled_operator.max_conn_size == 3


@pytest.mark.parametrize(
    "kind", ["conditional", "scatter", "permute", "dynamic_index", "symbol_default"]
)
def test_general_update_programs_and_dynamic_index_fallback(kind):
    hi = nk.hilbert.Fock(n_max=2, N=3)
    n = nkdsl.source_index(0)
    if kind == "conditional":
        update = nkdsl.Update.cond(n > 0, if_true=nkdsl.swap(0, 1), if_false=nkdsl.write(2, 1))
    elif kind == "scatter":
        update = nkdsl.scatter([0, 2], [nkdsl.source_index(2), n])
    elif kind == "permute":
        update = nkdsl.permute(0, 1, 2)
    elif kind == "dynamic_index":
        update = nkdsl.write(n, 1)
    else:
        update = nkdsl.shift(0, nkdsl.symbol("delta", default=1, dtype=np.int32))
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, kind)
        .globally()
        .emit(update, matrix_element=1.0)
        .emit(update, matrix_element=2.0)
        .emit(nkdsl.identity(), matrix_element=-1.0)
        .build()
    )
    artifact = _assert_equivalent(symbolic)
    assert artifact.compiled_operator.max_conn_size <= 2


def test_dead_branches_cancellation_and_complex_amplitudes():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "dead", dtype="complex128")
        .globally()
        .emit(nkdsl.write(0, 1), matrix_element=2 + 3j)
        .emit(nkdsl.write(0, 1), matrix_element=-2 - 3j)
        .emit(nkdsl.write(0, 2), matrix_element=1j)
        .emit(nkdsl.write(1, 1), matrix_element=0.0)
        .emit(nkdsl.identity().invalidate(), matrix_element=7.0)
        .for_each_site("i")
        .where(False)
        .emit(matrix_element=99.0)
        .build()
    )
    artifact = _assert_equivalent(symbolic)
    assert artifact.compiled_operator.max_conn_size == 1


def test_all_dead_operator_retains_one_zero_padding_slot():
    hi = nk.hilbert.Fock(n_max=1, N=2)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "zero")
        .for_each_site("i")
        .emit(matrix_element=0.0)
        .build()
    )
    artifact = _assert_equivalent(symbolic)
    assert artifact.compiled_operator.max_conn_size == 1


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.uint8, jnp.int32, jnp.float32, jnp.float64])
def test_integer_range_guard_handles_overflow_and_unsigned_inputs(dtype):
    hi = nk.hilbert.Fock(n_max=2, N=2)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "overflow")
        .globally()
        .emit(nkdsl.shift(0, 256), matrix_element=1.0)
        .emit(nkdsl.identity(), matrix_element=2.0)
        .emit(nkdsl.shift(1, -1), matrix_element=3.0)
        .build()
    )
    _assert_equivalent(symbolic, jnp.asarray(hi.all_states(), dtype=dtype))


def test_float_operand_precision_invalidates_integer_proofs():
    hi = nk.hilbert.HomogeneousHilbert(
        local_states=nk.utils.StaticRange(start=2**30, step=1, length=2, dtype=np.int64),
        N=2,
    )
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "promotion")
        .globally()
        .emit(nkdsl.shift(0, np.float32(1)), matrix_element=1.0)
        .emit(nkdsl.identity(), matrix_element=2.0)
        .build()
    )
    _assert_equivalent(symbolic, jnp.asarray(hi.all_states(), dtype=jnp.int64))


@pytest.mark.parametrize("size,width", [(1, 3), (13, 5), (40, 3), (65, 64)])
def test_general_runtime_algorithms_handle_duplicates_zero_and_invalid_branches(size, width):
    rng = np.random.default_rng(123)
    keys = rng.integers(-2, 3, size=(size, width)).astype(np.int32)
    if size > 3:
        keys[-3:] = keys[0]
    weights = rng.normal(size=size) + 1j * rng.normal(size=size)
    valids = rng.random(size) > 0.2
    xp, mels = map(
        np.asarray,
        jax.jit(coalesce_connected_components)(
            jnp.asarray(keys),
            jnp.asarray(weights),
            jnp.asarray(valids),
        ),
    )
    expected = {}
    for row, weight, valid in zip(keys, weights, valids):
        expected[tuple(row)] = expected.get(tuple(row), 0) + (weight if valid else 0)
    actual = {tuple(row): mel for row, mel in zip(xp, mels) if mel != 0}
    assert len(actual) == np.count_nonzero(mels)
    for key in expected.keys() | actual.keys():
        np.testing.assert_allclose(expected.get(key, 0), actual.get(key, 0), atol=1e-12)


def test_lossless_packing_crosses_word_boundaries_and_handles_negative_floats():
    ranges = ((-1, 1),) * 20 + ((0, 2**31 - 1),)
    rng = np.random.default_rng(15)
    values = rng.integers(-1, 2, size=(100, 21)).astype(np.float64)
    values[:, -1] = rng.integers(0, 2**31, size=100)
    keys = np.asarray(jax.jit(lambda x: pack_integer_keys(x, ranges))(jnp.asarray(values)))
    assert keys.shape == (100, 3)
    for a in range(100):
        for b in range(a):
            assert np.array_equal(keys[a], keys[b]) == np.array_equal(values[a], values[b])


def test_compaction_empty_single_and_reduced_capacity():
    xp, mels = compact_connections(jnp.zeros((0, 2)), jnp.zeros((0,)), output_size=1)
    assert xp.shape == (1, 2) and mels.shape == (1,)
    xp, mels = compact_connections(jnp.array([[7, 8]]), jnp.array([0.0]))
    np.testing.assert_array_equal(xp, [[0, 0]])
    xp, mels = jax.jit(lambda x, m: compact_connections(x, m, 2))(
        jnp.arange(10).reshape(5, 2),
        jnp.array([0, 2, 0, 3, 0]),
    )
    np.testing.assert_array_equal(xp, [[2, 3], [6, 7]])
    np.testing.assert_array_equal(mels, [2, 3])


def test_batched_shapes_and_amplitude_gradients():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "grad")
        .for_each_site("i")
        .emit(matrix_element=nkdsl.site("i").value.pow(2))
        .build()
    )
    compiled = _compile(symbolic).compiled_operator
    values = jnp.array([1.0, 2.0])
    np.testing.assert_allclose(
        jax.grad(lambda x: jnp.sum(compiled.get_conn_padded(x)[1]))(values), [2, 4]
    )
    xp, mels = compiled.get_conn_padded(jnp.ones((2, 3, 2)))
    assert xp.shape == (2, 3, 1, 2) and mels.shape == (2, 3, 1)
    xp, mels = jax.vmap(compiled.get_conn_padded)(jnp.ones((3, 2)))
    assert xp.shape == (3, 1, 2) and mels.shape == (3, 1)


def _cost_context(**options):
    hi = nk.hilbert.Fock(n_max=2, N=3)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "cost")
        .for_each_site("i")
        .emit(nkdsl.write(0, nkdsl.site("i").value), matrix_element=1.0)
        .build()
    )
    context = SymbolicCompilationContext(
        operator=symbolic,
        ir=symbolic.to_ir(),
        options=nkdsl.SymbolicCompilerOptions(cache_enabled=False, **options),
    )
    SymbolicMaxConnSizeAnalysisPass().run(context)
    SymbolicDeduplicationPass().run(context)
    return context


def test_cost_warning_and_structured_diagnostics():
    context = _cost_context()
    with pytest.warns(UserWarning, match="NKDSL-W401"):
        result = SymbolicConnectivityCostPass().run(context)
    assert result["estimated_assembly_ratio"] > 1
    assert context.analysis("dsl_diagnostics")[-1]["code"] == "NKDSL-W401"
    assert context.analysis("dsl_diagnostics_summary")["visible_warning"] == 1


@pytest.mark.parametrize(
    "options",
    [
        {"diagnostics_enabled": False},
        {"diagnostics_min_severity": "error"},
        {"deduplicate_connected_components": False},
    ],
)
def test_cost_warning_respects_diagnostics_and_dedup_options(options):
    context = _cost_context(**options)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        SymbolicConnectivityCostPass().run(context)
    assert not recorded


def test_cost_warning_can_fail_compilation():
    context = _cost_context(fail_on_warnings=True)
    with pytest.raises(SymbolicDiagnosticsError, match="NKDSL-W401"):
        SymbolicConnectivityCostPass().run(context)
    with pytest.raises(SymbolicCompilerError, match="NKDSL-W401"):
        nkdsl.SymbolicCompiler(options=context.options).compile(context.operator)


def test_static_distinct_constant_branches_need_no_extra_assembly_or_warning():
    hi = nk.hilbert.Spin(s=0.5, N=4)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "distinct")
        .for_each_site("i")
        .emit(nkdsl.write("i", -nkdsl.site("i").value), matrix_element=1)
        .build()
    )
    artifact = _assert_equivalent(symbolic)
    assert artifact.metadata_map()["connectivity_cost"]["estimated_assembly_ratio"] == 1
    assert not artifact.metadata_map()["connectivity_deduplication"]["compact"]


def test_analysis_budget_exhaustion_retains_exact_fallback(monkeypatch):
    import nkdsl.compiler.passes.deduplication as analysis

    monkeypatch.setattr(analysis, "_MAX_PROOF_ASSIGNMENTS", 0)
    monkeypatch.setattr(analysis, "_MAX_COLLISION_PAIRS", 0)
    hi = nk.hilbert.Fock(n_max=2, N=3)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "budget")
        .for_each_site("i")
        .emit(nkdsl.swap(0, "i"), matrix_element=1.0)
        .build()
    )
    artifact = _assert_equivalent(symbolic)
    assert artifact.metadata_map()["connectivity_deduplication"]["packed_key_widths"] == (1,)


def test_large_planned_collision_component_uses_single_sort_and_reduced_capacity():
    hi = nk.hilbert.HomogeneousHilbert(
        local_states=nk.utils.StaticRange(start=0, step=1024, length=16, dtype=np.int32),
        N=129,
    )
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "large-copy-image")
        .for_each_site("i")
        .emit(nkdsl.write(0, nkdsl.site("i").value), matrix_element=1.0)
        .build()
    )
    inputs = 1024 * jax.random.randint(jax.random.key(99), (32, 129), minval=0, maxval=16)
    artifact = _assert_equivalent(symbolic, inputs)
    assert artifact.compiled_operator.max_conn_size == 16
    assert artifact.metadata_map()["connectivity_cost"]["runtime_strategies"] == ("sort",)
    hlo = (
        jax.jit(artifact.compiled_operator.get_conn_padded)
        .lower(inputs)
        .compiler_ir(dialect="hlo")
        .as_hlo_text()
    )
    assert sum(" sort(" in line for line in hlo.splitlines()) == 1


@pytest.mark.parametrize("seed", range(5))
def test_mixed_generated_update_programs_match_direct_branch_aggregation(seed):
    rng = np.random.default_rng(seed)
    hi = nk.hilbert.Fock(n_max=2, N=3)
    builder = nkdsl.SymbolicDiscreteJaxOperator(hi, f"mixed-{seed}").globally()
    for _ in range(8):
        update = nkdsl.identity()
        for _ in range(3):
            a, b = map(int, rng.integers(0, 3, size=2))
            choice = int(rng.integers(0, 5))
            if choice == 0:
                update = update.swap(a, b)
            elif choice == 1:
                update = update.write(a, nkdsl.source_index(b))
            elif choice == 2:
                update = update.shift(a, int(rng.integers(-1, 2)))
            elif choice == 3:
                update = update.affine(a, scale=-1, bias=2)
            else:
                update = update.permute(0, 1, 2)
        builder = builder.emit(update, matrix_element=int(rng.integers(-3, 4)))
    _assert_equivalent(builder.build())


def test_float_local_basis_keeps_general_exact_collision_checks():
    hi = nk.hilbert.HomogeneousHilbert(
        local_states=nk.utils.StaticRange(start=-0.5, step=0.5, length=3),
        N=2,
    )
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "fractional-basis")
        .globally()
        .emit(nkdsl.swap(0, 1), matrix_element=1.0)
        .emit(nkdsl.write(0, 0.5), matrix_element=2.0)
        .emit(nkdsl.identity(), matrix_element=3.0)
        .build()
    )
    _assert_equivalent(symbolic)


def test_config_dtypes_do_not_change_proven_copy_capacity():
    hi = nk.hilbert.Fock(n_max=1, N=3)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "copy-dtypes")
        .for_each_site("i")
        .emit(nkdsl.write(0, nkdsl.site("i").value), matrix_element=1.0)
        .build()
    )
    _assert_equivalent(symbolic, jnp.asarray(hi.all_states(), dtype=jnp.complex64))


def test_zero_width_runtime_keys_and_nan_targets():
    xp, mels = jax.jit(coalesce_connected_components)(
        jnp.zeros((129, 0)),
        jnp.ones((129,)),
        jnp.ones((129,), dtype=bool),
    )
    assert xp.shape == (129, 0)
    np.testing.assert_allclose(mels, np.r_[129.0, np.zeros(128)])
    xp, mels = jax.jit(coalesce_connected_components)(
        jnp.array([[jnp.nan], [jnp.nan], [1.0], [1.0]]),
        jnp.ones((4,)),
        jnp.ones((4,), dtype=bool),
    )
    np.testing.assert_allclose(mels, [1, 1, 2, 0])


def test_constant_predicate_rounding_cannot_remove_a_live_branch():
    hi = nk.hilbert.Fock(n_max=1, N=2)
    large = nkdsl.AmplitudeExpr.constant(np.float32(2**24))
    predicate = ((large + np.float32(1)) - large) == 0
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "rounded-predicate")
        .globally()
        .where(predicate)
        .emit(matrix_element=1.0)
        .build()
    )
    _assert_equivalent(symbolic)


def test_opaque_programs_ignore_unused_iterator_bindings_for_value_numbering():
    hi = nk.hilbert.Fock(n_max=2, N=3)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "opaque-equivalent")
        .for_each_site("i")
        .emit(nkdsl.write(nkdsl.source_index(0), 1), matrix_element=1.0)
        .build()
    )
    artifact = _assert_equivalent(symbolic)
    assert artifact.compiled_operator.max_conn_size == 1


def test_assembly_plan_is_immutable_deterministic_and_cacheable():
    hi = nk.hilbert.Fock(n_max=2, N=3)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "plan-cache")
        .for_each_site("i")
        .emit(matrix_element=nkdsl.site("i").value)
        .build()
    )
    plan = analyze_connectivity(symbolic.to_ir(), hi)
    assert plan == analyze_connectivity(symbolic.to_ir(), hi)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.output_size = 99
    compiler = nkdsl.SymbolicCompiler(
        options=nkdsl.SymbolicCompilerOptions(diagnostics_enabled=False)
    )
    first = compiler.compile(symbolic)
    second = compiler.compile(symbolic)
    assert first is second
    assert second.metadata_map()["connectivity_deduplication"] == plan.as_dict()


def test_affine_disjointness_does_not_require_enumerating_a_large_local_basis():
    hi = nk.hilbert.Fock(n_max=500, N=3)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "large-local-basis")
        .globally()
        .emit(nkdsl.identity(), matrix_element=1.0)
        .for_each_site("i")
        .emit(nkdsl.shift("i", 1), matrix_element=1.0)
        .emit(nkdsl.shift("i", -1), matrix_element=1.0)
        .build()
    )
    inputs = jax.random.randint(jax.random.key(7), (32, 3), minval=0, maxval=501, dtype=jnp.int32)
    artifact = _assert_equivalent(symbolic, inputs)
    plan = artifact.metadata_map()["connectivity_deduplication"]
    assert plan["runtime_component_sizes"] == ()
    assert not plan["compact"]


def test_affine_normalization_tracks_overflowing_original_intermediates():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "affine-overflow")
        .globally()
        .emit(nkdsl.identity(), matrix_element=1.0)
        .emit(nkdsl.affine(0, scale=256, bias=0).shift(0, 1), matrix_element=2.0)
        .build()
    )
    _assert_equivalent(symbolic, jnp.asarray(hi.all_states(), dtype=jnp.int8))


def test_deduplication_pass_preserves_bounds_for_non_symbolic_ir():
    context = _cost_context()
    context = SymbolicCompilationContext(
        operator=context.operator,
        ir=dataclasses.replace(context.ir, mode="jax_kernel"),
        options=context.options,
    )
    context.set_analysis("total_max_conn_size", 17)
    assert SymbolicDeduplicationPass().run(context) == {"enabled": False}
    assert context.analysis("total_max_conn_size") == 17
    assert context.analysis("connectivity_deduplication_plan") is None


def test_colliding_component_and_disjoint_target_share_one_assembly_kernel():
    hi = nk.hilbert.Fock(n_max=2, N=3)
    symbolic = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "partial-component")
        .globally()
        .emit(nkdsl.write(0, nkdsl.source_index(1)), matrix_element=1.0)
        .emit(nkdsl.write(0, nkdsl.source_index(2)), matrix_element=2.0)
        .emit(nkdsl.shift_mod(2, 1), matrix_element=3.0)
        .build()
    )
    artifact = _assert_equivalent(symbolic)
    assert artifact.metadata_map()["connectivity_deduplication"]["runtime_component_sizes"] == (2,)
