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

"""Kernel-plan semantics against independent finite-state and JAX references."""

from __future__ import annotations

import dataclasses
import itertools
import re

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

import nkdsl as dsl
from nkdsl.compiler.core.actions import _constant
from nkdsl.compiler.core.kernel import scalar_template
from nkdsl.compiler.lowering.kernel_lowerer import _unit_wrap_legal
from nkdsl.compiler.passes import kernel_planning

pytestmark = pytest.mark.unit


def compile_artifact(op, optimize=True, dedup=True, **options):
    return dsl.SymbolicCompiler(
        options=dsl.SymbolicCompilerOptions(
            optimize_kernels=optimize,
            deduplicate_connected_components=dedup,
            diagnostics_enabled=False,
            cache_enabled=False,
            **options,
        )
    ).compile(op)


def compare_reference(op, x, dedup=True):
    optimized = compile_artifact(op, dedup=dedup)
    baseline = compile_artifact(op, optimize=False, dedup=dedup)
    actual = jax.block_until_ready(jax.jit(optimized.compiled_operator.get_conn_padded)(x))
    expected = jax.block_until_ready(jax.jit(baseline.compiled_operator.get_conn_padded)(x))
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-6, atol=1e-12, equal_nan=True)
    assert optimized.compiled_operator.max_conn_size == baseline.compiled_operator.max_conn_size
    return optimized, actual


def independent_connections(source, rows, q):
    entries = {}
    for i, j in rows:
        target = source.copy()
        target[i] = (target[i] + 1) % q
        target[j] = (target[j] - 1) % q
        weight = np.sqrt(source[i] + 1) * (1 + target[j] ** 2)
        key = tuple(target)
        entries[key] = entries.get(key, 0) + weight
    return {key: value for key, value in entries.items() if value != 0}


@pytest.mark.parametrize("state_dtype", [np.int8, np.int32, np.float32, np.float64])
@pytest.mark.parametrize("dedup", [False, True])
def test_irregular_aliased_moves_and_nonlinear_emitted_amplitude(state_dtype, dedup):
    hi = nk.hilbert.Fock(n_max=2, N=3)
    rows = [(0, 1), (2, 0), (1, 2), (0, 1), (1, 1)]
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "unrelated_name")
        .for_each(("i", "j"), over=rows)
        .emit(
            dsl.shift_mod("i", 1).shift_mod("j", -1),
            matrix_element=(dsl.site("i").value + 1).sqrt() * (1 + dsl.emitted("j").value.pow(2)),
        )
        .build()
    )
    sources = np.asarray(hi.all_states(), dtype=state_dtype)
    artifact, (targets, weights) = compare_reference(op, jnp.asarray(sources), dedup)
    for source, row, mels in zip(sources, np.asarray(targets), np.asarray(weights)):
        actual = {}
        for target, weight in zip(row, mels):
            if weight:
                key = tuple(target)
                actual[key] = actual.get(key, 0) + weight
        expected = independent_connections(source.astype(np.int64), rows, 3)
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6)
    assert artifact.metadata_map()["kernel_planning"]["enabled"]


@pytest.mark.parametrize(
    "kind",
    [
        "diagonal",
        "affine",
        "write",
        "scatter",
        "permutation",
        "conditional",
        "dynamic",
        "long_program",
        "complex",
    ],
)
@pytest.mark.parametrize("dedup", [False, True])
def test_general_program_families(kind, dedup):
    hi = nk.hilbert.Fock(n_max=2, N=3)
    a, b = dsl.source_index(0), dsl.source_index(1)
    updates = {
        "diagonal": dsl.identity(),
        "affine": dsl.affine(0, scale=-1, bias=2),
        "write": dsl.write(0, (a - b).abs_().sqrt()),
        "scatter": dsl.scatter([0, 2], [b, a]),
        "permutation": dsl.permute(0, 1, 2),
        "conditional": dsl.Update.cond(a > 0, if_true=dsl.swap(0, 1), if_false=dsl.write(2, 1)),
        "dynamic": dsl.write(a, b + 1),
        "long_program": dsl.shift(0, 1).write(1, a).swap(0, 2).shift(0, -1),
        "complex": dsl.swap(0, 1),
    }
    amp = (a + dsl.target_index(0) + 1).sqrt()
    if kind == "complex":
        amp = (amp + 1j * dsl.target_index(1)).conj()
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, kind)
        .globally()
        .emit(updates[kind], matrix_element=amp)
        .emit(matrix_element=a - b)
        .build()
    )
    artifact, _ = compare_reference(op, jnp.asarray(hi.all_states()), dedup)
    meta = artifact.metadata_map()["kernel_planning"]
    assert meta["opaque_targets"] == (1 if kind == "dynamic" else 0)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 3), (0, 3)])
def test_batch_leading_dimensions_and_zero_batch(shape):
    hi = nk.hilbert.Fock(n_max=2, N=3)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "batch")
        .for_each_site("i")
        .emit(dsl.shift_mod("i", 1), matrix_element=-0.7)
        .build()
    )
    x = jnp.ones(shape, dtype=jnp.int32)
    _, (xp, mels) = compare_reference(op, x)
    assert xp.shape == shape[:-1] + (3, 3)
    assert mels.shape == shape[:-1] + (3,)


@pytest.mark.parametrize("mapped", [False, True])
def test_masked_singular_amplitude_and_derivative(mapped):
    hi = nk.hilbert.Fock(n_max=2, N=1)
    b = dsl.SymbolicDiscreteJaxOperator(hi, "guard")
    b = b.for_each_site("i") if mapped else b.globally()
    a = dsl.site("i").value if mapped else dsl.source_index(0)
    op = b.emit_if(a > 0, matrix_element=a.sqrt()).emit_else(matrix_element=2.0).build()
    x = jnp.asarray([-1.0])
    funcs = [
        compile_artifact(op, optimize=v).compiled_operator.get_conn_padded for v in [False, True]
    ]
    for fn in funcs:
        np.testing.assert_allclose(fn(x)[1], [2.0])
    grads = [jax.grad(lambda s: jnp.sum(fn(s)[1]))(x) for fn in funcs]
    np.testing.assert_allclose(grads[0], grads[1], equal_nan=True)


def test_unmapped_conditional_target_preserves_finite_inactive_derivative():
    hi = nk.hilbert.Fock(n_max=2, N=1)
    a = dsl.source_index(0)
    update = dsl.Update.cond(a > 0, if_true=dsl.write(0, a.sqrt()), if_false=dsl.write(0, 2))
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "target_region")
        .globally()
        .emit(update, matrix_element=dsl.target_index(0))
        .build()
    )
    derivatives = []
    for optimize in [False, True]:
        fn = compile_artifact(op, optimize, dedup=False).compiled_operator.get_conn_padded
        derivatives.append(jax.grad(lambda x: sum(jnp.sum(v) for v in fn(x)))(jnp.array([-1.0])))
    np.testing.assert_array_equal(derivatives[0], [0.0])
    np.testing.assert_array_equal(derivatives[1], derivatives[0])


def test_scalar_constants_keep_weak_promotion_and_signed_zero_in_raw_mode():
    hi = nk.hilbert.Fock(n_max=1, N=2)
    a = dsl.source_index(0)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "typed", dtype="float64")
        .globally()
        .emit(matrix_element=1.23456789012345 + a)
        .emit(matrix_element=np.float64(1.23456789012345) + a)
        .emit(matrix_element=0.0)
        .emit(matrix_element=-0.0)
        .build()
    )
    _, (_, weights) = compare_reference(op, jnp.zeros((2, 2), jnp.float32), False)
    assert np.signbit(np.asarray(weights)[0, -1])
    assert not np.signbit(np.asarray(weights)[0, -2])


def test_dense_table_budget_selects_sparse_schedule_without_changing_results():
    op = cyclic_operator(6)
    compiler = dsl.SymbolicCompiler(options=dsl.SymbolicCompilerOptions(diagnostics_enabled=False))
    artifact = compiler.compile(op)
    from nkdsl.compiler.core.context import SymbolicCompilationContext
    from nkdsl.compiler.passes.analysis import SymbolicMaxConnSizeAnalysisPass
    from nkdsl.compiler.passes.deduplication import SymbolicDeduplicationPass
    from nkdsl.compiler.lowering.kernel_lowerer import make_kernel_function, kernel_schedule_summary

    context = SymbolicCompilationContext(operator=op, ir=op.to_ir(), options=compiler.options)
    for p in [
        SymbolicMaxConnSizeAnalysisPass(),
        SymbolicDeduplicationPass(),
        kernel_planning.SymbolicKernelPlanningPass(),
    ]:
        p.run(context)
    plan = dataclasses.replace(context.require_analysis("kernel_plan"), dense_table_limit=0)
    assert kernel_schedule_summary(plan, 6)["target_strategy"] == "sparse_scatter"
    x = jnp.asarray(op.hilbert.random_state(jax.random.key(1), size=16))
    actual = make_kernel_function(
        plan, 6, np.dtype("float64"), artifact.compiled_operator.max_conn_size
    )(x)
    expected = artifact.compiled_operator.get_conn_padded(x)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def cyclic_operator(size=18):
    hi = nk.hilbert.Fock(n_max=4, N=size)
    rows = [tuple((2 * p + k) % size for k in range(4)) for p in range(size // 2)]
    forward = dsl.shift_mod("a", 1).shift_mod("b", 1).shift_mod("c", -1).shift_mod("d", -1)
    reverse = dsl.shift_mod("a", -1).shift_mod("b", -1).shift_mod("c", 1).shift_mod("d", 1)
    return (
        dsl.SymbolicDiscreteJaxOperator(hi, "static_cyclic")
        .for_each_site("i")
        .emit(matrix_element=0.5 * (dsl.site("i").value - 2).pow(2))
        .for_each(("a", "b", "c", "d"), over=rows)
        .emit(matrix_element=0.5)
        .emit(forward, matrix_element=-0.25)
        .emit(reverse, matrix_element=-0.25)
        .build()
    )


def hlo_ops(text):
    return re.findall(r" = [^\n]*?\b([a-z][a-z-]*)\(", text)


def test_static_cyclic_lowering_has_no_generic_assembly_instructions():
    op = cyclic_operator()
    artifact = compile_artifact(op)
    x = jnp.zeros((16, 18), dtype=jnp.int64)
    kernel = jax.jit(artifact.compiled_operator.get_conn_padded).lower(x)
    ops = hlo_ops(kernel.compiler_ir(dialect="hlo").as_hlo_text())
    assert not {"scatter", "sort", "remainder", "reduce-window"}.intersection(ops)
    summary = artifact.metadata_map()["kernel_planning"]
    assert summary["target_strategy"] == "dense_select"
    assert summary["scalar_batches"] < summary["raw_actions"]
    compare_reference(op, x)


@pytest.mark.parametrize("q", [1, 2, 3, 5])
def test_repeated_inverse_and_identity_modular_actions(q):
    hi = nk.hilbert.Fock(n_max=q - 1, N=2)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "period")
        .for_each_site("i")
        .emit(dsl.shift_mod("i", 1), matrix_element=0.25)
        .emit(dsl.shift_mod("i", -1), matrix_element=-0.5)
        .emit(dsl.shift_mod("i", q), matrix_element=0.75)
        .build()
    )
    compare_reference(op, jnp.asarray(hi.all_states()))


@pytest.mark.parametrize("value", [0.0, -0.0, 1e-45, float("inf"), float("nan")])
def test_zero_underflow_and_nonfinite_constants_keep_padding_contract(value):
    hi = nk.hilbert.Fock(n_max=1, N=2)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "weights", dtype="float32")
        .for_each_site("i")
        .emit(dsl.shift_mod("i", 1), matrix_element=value)
        .build()
    )
    compare_reference(op, jnp.asarray(hi.all_states()))


def test_collision_can_create_zero_slots_from_nonzero_inputs():
    hi = nk.hilbert.Fock(n_max=1, N=3)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "collisions")
        .for_each_site("i")
        .emit(dsl.write(0, dsl.site("i").value), matrix_element=0.25)
        .build()
    )
    _, (xp, weights) = compare_reference(op, jnp.asarray(hi.all_states()))
    for row, mels in zip(np.asarray(xp), np.asarray(weights)):
        nonzero = np.flatnonzero(mels)
        np.testing.assert_array_equal(nonzero, np.arange(len(nonzero)))
        assert np.all(row[len(nonzero) :] == 0)


def test_constant_templates_distinguish_signed_zero_and_read_aliasing():
    assert _constant(0.0) != _constant(-0.0)
    template, indices = scalar_template((("add", ("read", 2), ("read", 2)),))
    assert indices == (2,)
    assert template == (("add", ("input", 0), ("input", 0)),)
    other, _ = scalar_template((("add", ("read", 2), ("read", 3)),))
    assert template != other


@pytest.mark.parametrize(
    "dtype, legal", [(np.int8, False), (np.int16, True), (np.float32, True), (np.complex64, False)]
)
def test_unit_wrap_proof_checks_every_original_intermediate(dtype, legal):
    basis = tuple(range(-64, 64))
    assert _unit_wrap_legal(np.dtype(dtype), (-64, 128), 1, basis) == legal


@pytest.mark.parametrize("budget", ["_MAX_ACTIONS", "_MAX_SCALAR_BATCHES"])
def test_planning_budget_keeps_reference_fallback(monkeypatch, budget):
    monkeypatch.setattr(kernel_planning, budget, 0)
    artifact, _ = compare_reference(cyclic_operator(6), jnp.zeros((2, 6), jnp.int32))
    assert "budget_exceeded" in artifact.metadata_map()["kernel_planning"]["fallback_reason"]


def test_option_override_and_cache_signature_are_preserved():
    op = cyclic_operator(6)
    options = dsl.SymbolicCompilerOptions(diagnostics_enabled=False, optimize_kernels=False)
    compiler = dsl.SymbolicCompiler(options=options, cache_enabled=True)
    first = compiler.compile(op)
    assert not first.metadata_map()["kernel_planning"]["enabled"]
    assert compiler.compile(op) is first
    second = compiler.compile(op, options=dataclasses.replace(options, optimize_kernels=True))
    assert second.cache_key != first.cache_key
    assert second.metadata_map()["kernel_planning"]["enabled"]
    overridden = op.compile(compiler=compiler, optimize_kernels=True)
    compare = second.compiled_operator
    np.testing.assert_array_equal(
        overridden.get_conn_padded(jnp.zeros(6))[0], compare.get_conn_padded(jnp.zeros(6))[0]
    )


def test_iterator_rows_participate_in_source_ir_fingerprint():
    hi = nk.hilbert.Fock(n_max=2, N=3)

    def build(rows):
        return (
            dsl.SymbolicDiscreteJaxOperator(hi, "same_name")
            .for_each(("a", "b"), over=rows)
            .emit(dsl.swap("a", "b"), matrix_element=1)
            .build()
        )

    assert (
        build([(0, 1)]).to_ir().static_fingerprint() != build([(1, 2)]).to_ir().static_fingerprint()
    )


def test_iterator_index_arithmetic_retains_int32_overflow_and_emitted_index():
    hi = nk.hilbert.Fock(n_max=1, N=4)
    index = dsl.site("i").index
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "index_dtype")
        .for_each_site("i")
        .emit(dsl.write(0, index.pow(31)), matrix_element=index.pow(31) + dsl.emitted("i").index)
        .build()
    )
    _, (_, weights) = compare_reference(op, jnp.zeros((2, 4), dtype=jnp.int64), False)
    assert np.asarray(weights)[0, 2] == -2147483646


def test_weak_configuration_and_x64_specializations_keep_reference_results():
    hi = nk.hilbert.Fock(n_max=1, N=2)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "specializations", dtype="float32")
        .for_each_site("i")
        .emit(dsl.shift_mod("i", 1), matrix_element=-0.25)
        .build()
    )
    original = bool(getattr(jax.config, "jax_enable_x64"))
    try:
        for enabled in [True, False]:
            jax.config.update("jax_enable_x64", enabled)
            compare_reference(op, jnp.broadcast_to(jnp.asarray(1.0), (2, 2)), True)
    finally:
        jax.config.update("jax_enable_x64", original)


@pytest.mark.parametrize("index", [-1, -2, 5])
def test_emitted_read_index_maps_match_reference_gather_semantics(index):
    hi = nk.hilbert.Fock(n_max=2, N=3)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "read_maps")
        .for_each(("i",), over=[(index,)])
        .emit(dsl.write(2, 1), matrix_element=dsl.emitted("i").value)
        .build()
    )
    compare_reference(op, jnp.asarray(hi.all_states()))


@pytest.mark.parametrize("shape", [(), (2,), (2, 4)])
def test_invalid_source_shapes_fail_clearly(shape):
    fn = compile_artifact(cyclic_operator(6)).compiled_operator.get_conn_padded
    with pytest.raises(ValueError, match="final axis"):
        fn(jnp.zeros(shape))


@pytest.mark.parametrize("seed", range(5))
def test_generated_mixed_programs_compare_both_dtypes_and_modes(seed):
    rng = np.random.default_rng(seed)
    hi = nk.hilbert.Fock(n_max=2, N=3)
    b = dsl.SymbolicDiscreteJaxOperator(hi, "generated").globally()
    for _ in range(5):
        i, j = map(int, rng.integers(0, 3, 2))
        u = dsl.shift_mod(i, int(rng.choice([-1, 1]))).swap(i, j)
        a = dsl.source_index(i)
        b = b.emit(u, matrix_element=(a + dsl.target_index(j) + 1).sqrt())
    op = b.build()
    for dtype, dedup in itertools.product([jnp.int8, jnp.float32], [False, True]):
        compare_reference(op, jnp.asarray(hi.all_states(), dtype=dtype), dedup)


def planned_kernel(op, **changes):
    from nkdsl.compiler.core.context import SymbolicCompilationContext
    from nkdsl.compiler.passes.analysis import SymbolicMaxConnSizeAnalysisPass
    from nkdsl.compiler.passes.deduplication import SymbolicDeduplicationPass
    from nkdsl.compiler.lowering.kernel_lowerer import make_kernel_function

    context = SymbolicCompilationContext(
        operator=op, ir=op.to_ir(), options=dsl.SymbolicCompilerOptions()
    )
    for p in [
        SymbolicMaxConnSizeAnalysisPass(),
        SymbolicDeduplicationPass(),
        kernel_planning.SymbolicKernelPlanningPass(),
    ]:
        p.run(context)
    plan = dataclasses.replace(context.require_analysis("kernel_plan"), **changes)
    artifact = compile_artifact(op)
    return plan, make_kernel_function(
        plan,
        op.hilbert.size,
        artifact.compiled_operator.dtype,
        artifact.compiled_operator.max_conn_size,
    )


@pytest.mark.parametrize("state_dtype", [jnp.int32, jnp.float32])
@pytest.mark.parametrize("leading", [(64,), (2, 32)])
def test_large_guarded_target_dependent_batches_preserve_packing(state_dtype, leading):
    hi = nk.hilbert.Fock(n_max=4, N=12)
    a, b = dsl.site("i").value, dsl.site("j").value
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "late_pack")
        .for_each(("i", "j"), over=[(i, (i + 1) % 12) for i in range(12)])
        .where((a > 0) & (b < 4))
        .emit(dsl.shift("i", -1).shift("j", 1), matrix_element=a + dsl.emitted("j").value)
        .build()
    )
    x = jnp.asarray(np.random.default_rng(1).integers(0, 5, leading + (12,)), state_dtype)
    x = x.at[0].set(0)
    _, packed = planned_kernel(op)
    expected = compile_artifact(op, optimize=False).compiled_operator.get_conn_padded(x)
    for actual, reference in zip(packed(x), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-6)
    # An all-inactive row must have both zero weights and zero padded states.
    assert np.all(np.asarray(packed(x)[0])[0] == 0)


def test_sparse_copy_schedule_matches_irregular_copy_and_collision_semantics():
    hi = nk.hilbert.Fock(n_max=2, N=5)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "copy_maps")
        .for_each(("i", "j"), over=[(0, 3), (4, 1), (1, 3), (0, 3)])
        .emit(dsl.swap("i", "j"), matrix_element=dsl.emitted("i").value + 1)
        .emit(dsl.write("i", dsl.site("j").value), matrix_element=-0.25)
        .build()
    )
    artifact, _ = compare_reference(op, jnp.asarray(hi.all_states()))
    assert artifact.metadata_map()["kernel_planning"]["target_strategy"] == "sparse_scatter"


@pytest.mark.parametrize("delta", [-1, 1])
@pytest.mark.parametrize("q", [1, 2, 5])
def test_unit_wrap_preserves_reference_floating_jvp_and_vjp_at_endpoints(delta, q):
    hi = nk.hilbert.Fock(n_max=q - 1, N=2)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "wrap_derivative")
        .for_each_site("i")
        .emit(dsl.shift_mod("i", delta), matrix_element=1 + dsl.emitted("i").value)
        .build()
    )
    x = jnp.asarray([0.0, q - 1.0])
    reference = compile_artifact(op, optimize=False).compiled_operator.get_conn_padded
    optimized = compile_artifact(op).compiled_operator.get_conn_padded
    for fn in (lambda f: f(x), lambda f: jax.jacfwd(f)(x)):
        for actual, expected in zip(fn(optimized), fn(reference)):
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def total(fn, s):
        targets, weights = fn(s)
        return jnp.sum(targets) + jnp.sum(weights)

    np.testing.assert_allclose(
        jax.grad(lambda s: total(optimized, s))(x),
        jax.grad(lambda s: total(reference, s))(x),
        rtol=1e-6,
    )


def test_dense_nonlinear_target_never_speculates_on_unused_source_coordinates():
    hi = nk.hilbert.Fock(n_max=2, N=2)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "restricted_nonlinear")
        .globally()
        .emit(dsl.write(0, dsl.source_index(0).sqrt()), matrix_element=dsl.target_index(0) + 1)
        .emit(dsl.write(1, 2), matrix_element=1)
        .build()
    )
    # The target sqrt applies only to coordinate 0. Coordinate 1 is outside its
    # domain, but is a legal untouched source value for this local JAX check.
    x = jnp.asarray([1.0, -1.0])
    funcs = [
        compile_artifact(op, optimize=v, dedup=False).compiled_operator.get_conn_padded
        for v in [False, True]
    ]
    for projection in [lambda out: jnp.sum(out[0]), lambda out: jnp.sum(out[1])]:
        gradients = [jax.grad(lambda s: projection(fn(s)))(x) for fn in funcs]
        np.testing.assert_allclose(gradients[1], gradients[0])
        assert np.all(np.isfinite(gradients[1]))


@pytest.mark.parametrize("state_dtype", [jnp.int8, jnp.float32, jnp.float64])
@pytest.mark.parametrize("declared_dtype", [None, "float32", "int32"])
def test_default_symbol_casts_match_reference_promotion(state_dtype, declared_dtype):
    hi = nk.hilbert.Fock(n_max=2, N=2)
    coupling = dsl.symbol(
        "g", default=0.1 if declared_dtype != "int32" else 3, dtype=declared_dtype
    )
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "declared_defaults")
        .for_each_site("i")
        .emit(
            dsl.write("i", coupling + dsl.site("i").value),
            matrix_element=(coupling + dsl.site("i").value).pow(2),
        )
        .build()
    )
    compare_reference(op, jnp.asarray(hi.all_states(), state_dtype))


@pytest.mark.parametrize("dedup", [False, True])
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize(
    "amplitude",
    [
        dsl.AmplitudeExpr.constant(1e-20) * 1e-20,
        (dsl.AmplitudeExpr.constant(1e-20) * 1e-20) + 1e-20,
        dsl.AmplitudeExpr.constant(-1e-20) * 1e-20,
        (dsl.site("i").value * 1e-20) * 1e-20,
        dsl.AmplitudeExpr.constant(1e-320),
        dsl.AmplitudeExpr.constant(1e-20) / 1e20,
        dsl.AmplitudeExpr.constant(1e-20).pow(2),
        dsl.AmplitudeExpr.constant(float(np.float32(1.2e-38)))
        - float(np.nextafter(np.float32(1.2e-38), np.float32(0))),
    ],
)
def test_computed_subnormal_weights_preserve_zero_and_padding_semantics(amplitude, dedup, x64):
    hi = nk.hilbert.Fock(n_max=1, N=2)
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "computed_underflow", dtype="float32")
        .for_each_site("i")
        .emit(dsl.shift_mod("i", 1), matrix_element=amplitude)
        .build()
    )
    original = bool(getattr(jax.config, "jax_enable_x64"))
    try:
        jax.config.update("jax_enable_x64", x64)
        compare_reference(op, jnp.asarray(hi.all_states(), jnp.float32), dedup)
    finally:
        jax.config.update("jax_enable_x64", original)


@pytest.mark.parametrize("state_dtype", [jnp.int8, jnp.uint8, jnp.int32, jnp.int64, jnp.float32])
@pytest.mark.parametrize("exponent", [0, 1, 2, 3, 31, 63, 64, -1])
def test_constant_powers_preserve_typed_reference_overflow_and_boundaries(state_dtype, exponent):
    hi = nk.hilbert.Fock(n_max=4, N=2)
    a = dsl.site("i").value
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "integer_power")
        .for_each_site("i")
        .emit(dsl.write("i", a.pow(exponent)), matrix_element=(a + 1).pow(exponent))
        .build()
    )
    edge = [127, 255] if jnp.issubdtype(state_dtype, jnp.unsignedinteger) else [-3, -1]
    compare_reference(op, jnp.asarray([[0, 1], [2, 3], [4, 2], edge], state_dtype), dedup=False)


def test_integer_power_strength_reduction_preserves_strong_exponent_promotion():
    hi = nk.hilbert.Fock(n_max=4, N=2)
    a = dsl.site("i").value
    exponent = dsl.symbol("p", default=8, dtype="int32")
    op = (
        dsl.SymbolicDiscreteJaxOperator(hi, "power_promotion")
        .for_each_site("i")
        .emit(dsl.write("i", a.pow(exponent)), matrix_element=a.pow(exponent))
        .build()
    )
    compare_reference(op, jnp.asarray([[2, 3], [4, 2]], jnp.int8), dedup=False)


def test_integer_square_lowering_removes_dynamic_power_bit_operations():
    fn = compile_artifact(cyclic_operator()).compiled_operator.get_conn_padded
    lowered = jax.jit(fn).lower(jnp.zeros((16, 18), dtype=jnp.int64))
    ops = hlo_ops(lowered.compiler_ir(dialect="hlo").as_hlo_text())
    assert not {"shift-right-logical", "and", "power"}.intersection(ops)


def test_new_option_preserves_existing_positional_constructor_arguments():
    options = dsl.SymbolicCompilerOptions("jax", False, False)
    assert options.backend_preference == "jax"
    assert options.deduplicate_connected_components is False
    assert options.enable_fusion is False
    assert options.optimize_kernels is True
