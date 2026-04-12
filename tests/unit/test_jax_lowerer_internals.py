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

import types

import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

import nkdsl
from nkdsl.compiler.core.context import SymbolicCompilationContext
from nkdsl.compiler.core.options import SymbolicCompilerOptions
from nkdsl.compiler.lowering import jax_lowerer as jl
from nkdsl.ir.term import EmissionSpec
from nkdsl.ir.term import KBodyIteratorSpec
from nkdsl.ir.term import SymbolicIRTerm
from nkdsl.ir.update import UpdateOp

pytestmark = pytest.mark.unit


def _dummy_context_with_ir(ir, hi):
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "d", hermitian=True)
        .globally()
        .emit(nkdsl.identity(), matrix_element=1.0)
        .build()
    )
    return SymbolicCompilationContext(operator=op, ir=ir, options=SymbolicCompilerOptions())


def test_shift_mod_spec_inference_and_errors():
    hi_ok = nk.hilbert.Fock(n_max=2, N=1)
    state_min, mod_span = jl.infer_shift_mod_spec_from_hilbert(hi_ok)
    assert state_min == 0
    assert mod_span == 3

    with pytest.raises(ValueError, match="local_states"):
        jl.infer_shift_mod_spec_from_hilbert(types.SimpleNamespace(local_states=None))

    with pytest.raises(ValueError, match="non-empty 1D"):
        jl.infer_shift_mod_spec_from_hilbert(types.SimpleNamespace(local_states=np.zeros((2, 2))))

    with pytest.raises(ValueError, match="integer"):
        jl.infer_shift_mod_spec_from_hilbert(
            types.SimpleNamespace(local_states=np.asarray([0.0, 0.5, 1.0]))
        )

    with pytest.raises(ValueError, match="contiguous"):
        jl.infer_shift_mod_spec_from_hilbert(
            types.SimpleNamespace(local_states=np.asarray([0, 2, 3]))
        )


def test_eval_amplitude_and_predicate_all_main_paths():
    x = jnp.asarray([1, 2], dtype=jnp.int32)
    env = {
        "__x__": x,
        "__x_prime__": jnp.asarray([2, 3], dtype=jnp.int32),
        "site:i:value": x[0],
        "site:i:index": jnp.int32(0),
        "emit:i:value": jnp.int32(2),
        "emit:i:index": jnp.int32(0),
        "foo": 7,
    }

    expr = nkdsl.AmplitudeExpr.wrap_mod(
        nkdsl.AmplitudeExpr.abs_(
            nkdsl.AmplitudeExpr.pow(
                nkdsl.AmplitudeExpr.sqrt(
                    nkdsl.AmplitudeExpr.abs_(
                        nkdsl.AmplitudeExpr.conj(
                            nkdsl.AmplitudeExpr.neg(
                                nkdsl.AmplitudeExpr.add(
                                    nkdsl.AmplitudeExpr.symbol("foo"),
                                    nkdsl.AmplitudeExpr.sub(
                                        nkdsl.AmplitudeExpr.static_index(0),
                                        nkdsl.AmplitudeExpr.static_emitted_index(0),
                                    ),
                                )
                            )
                        )
                    )
                ),
                2,
            )
            / 3
        )
    )
    out = jl.eval_amplitude(expr, env, shift_mod_state_min=0, shift_mod_mod_span=3)
    assert np.isfinite(float(out))

    with pytest.raises(KeyError, match="not found"):
        jl.eval_amplitude(nkdsl.AmplitudeExpr.symbol("missing"), env)

    with pytest.raises(KeyError, match=r"requires env\['__x__'\]"):
        jl.eval_amplitude(nkdsl.AmplitudeExpr.static_index(0), {})

    with pytest.raises(KeyError, match=r"requires env\['__x_prime__'\]"):
        jl.eval_amplitude(nkdsl.AmplitudeExpr.static_emitted_index(0), {})

    with pytest.raises(ValueError, match="requires a resolved shift_mod_spec"):
        jl.eval_amplitude(nkdsl.AmplitudeExpr.wrap_mod(1), env)

    pred = nkdsl.PredicateExpr.and_(
        nkdsl.PredicateExpr.or_(
            nkdsl.PredicateExpr.eq(1, 1),
            nkdsl.PredicateExpr.ne(1, 2),
        ),
        nkdsl.PredicateExpr.and_(
            nkdsl.PredicateExpr.lt(0, 1),
            nkdsl.PredicateExpr.le(1, 1),
            nkdsl.PredicateExpr.gt(2, 1),
            nkdsl.PredicateExpr.ge(1, 1),
        ),
    )
    assert bool(jl.eval_predicate(pred, env)) is True
    assert bool(jl.eval_predicate(nkdsl.PredicateExpr.not_(False), env)) is True


def test_apply_single_update_op_variants():
    x = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    env = {
        "site:i:index": jnp.int32(1),
        "site:j:index": jnp.int32(2),
        "site:i:value": x[1],
        "site:j:value": x[2],
    }

    x1 = jl.apply_single_update_op(
        UpdateOp.from_mapping(
            kind="write_site",
            params={"site": nkdsl.site("i").index, "value": nkdsl.AmplitudeExpr.constant(5)},
        ),
        x,
        env,
        3,
    )
    np.testing.assert_array_equal(np.asarray(x1), np.asarray([0, 5, 2]))

    x2 = jl.apply_single_update_op(
        UpdateOp.from_mapping(
            kind="shift_site",
            params={"site": nkdsl.site("i").index, "delta": nkdsl.AmplitudeExpr.constant(2)},
        ),
        x,
        env,
        3,
    )
    np.testing.assert_array_equal(np.asarray(x2), np.asarray([0, 3, 2]))

    x3 = jl.apply_single_update_op(
        UpdateOp.from_mapping(
            kind="shift_mod_site",
            params={"site": nkdsl.site("i").index, "delta": nkdsl.AmplitudeExpr.constant(3)},
        ),
        x,
        env,
        3,
        shift_mod_state_min=0,
        shift_mod_mod_span=3,
    )
    np.testing.assert_array_equal(np.asarray(x3), np.asarray([0, 1, 2]))

    with pytest.raises(ValueError, match="without a resolved shift_mod_spec"):
        jl.apply_single_update_op(
            UpdateOp.from_mapping(
                kind="shift_mod_site",
                params={"site": nkdsl.site("i").index, "delta": nkdsl.AmplitudeExpr.constant(1)},
            ),
            x,
            env,
            3,
        )

    x4 = jl.apply_single_update_op(
        UpdateOp.from_mapping(
            kind="swap_sites",
            params={"site_a": nkdsl.site("i").index, "site_b": nkdsl.site("j").index},
        ),
        x,
        env,
        3,
    )
    np.testing.assert_array_equal(np.asarray(x4), np.asarray([0, 2, 1]))

    x5 = jl.apply_single_update_op(
        UpdateOp.from_mapping(
            kind="affine_site",
            params={
                "site": nkdsl.site("i").index,
                "scale": nkdsl.AmplitudeExpr.constant(2),
                "bias": nkdsl.AmplitudeExpr.constant(-1),
            },
        ),
        x,
        env,
        3,
    )
    np.testing.assert_array_equal(np.asarray(x5), np.asarray([0, 1, 2]))

    x6 = jl.apply_single_update_op(
        UpdateOp.from_mapping(
            kind="permute_sites",
            params={
                "sites": (
                    nkdsl.AmplitudeExpr.constant(0),
                    nkdsl.AmplitudeExpr.constant(1),
                    nkdsl.AmplitudeExpr.constant(2),
                )
            },
        ),
        x,
        env,
        3,
    )
    np.testing.assert_array_equal(np.asarray(x6), np.asarray([1, 2, 0]))

    x7 = jl.apply_single_update_op(
        UpdateOp.from_mapping(
            kind="scatter",
            params={
                "flat_indices": (0, 2),
                "values": (
                    nkdsl.AmplitudeExpr.constant(9),
                    nkdsl.AmplitudeExpr.constant(8),
                ),
            },
        ),
        x,
        env,
        3,
    )
    np.testing.assert_array_equal(np.asarray(x7), np.asarray([9, 1, 8]))

    cond = UpdateOp.from_mapping(
        kind="cond_branch",
        params={
            "predicate": nkdsl.PredicateExpr.constant(True),
            "then_ops": (
                UpdateOp.from_mapping(
                    kind="write_site",
                    params={
                        "site": nkdsl.AmplitudeExpr.constant(0),
                        "value": nkdsl.AmplitudeExpr.constant(4),
                    },
                ),
            ),
            "else_ops": (
                UpdateOp.from_mapping(
                    kind="write_site",
                    params={
                        "site": nkdsl.AmplitudeExpr.constant(0),
                        "value": nkdsl.AmplitudeExpr.constant(5),
                    },
                ),
            ),
        },
    )
    x8 = jl.apply_single_update_op(cond, x, env, 3)
    np.testing.assert_array_equal(np.asarray(x8), np.asarray([4, 1, 2]))


def test_apply_update_program_and_kbody_runner_shapes():
    hi = nk.hilbert.Fock(n_max=2, N=3)

    term_global = SymbolicIRTerm.create(
        name="g",
        iterator=KBodyIteratorSpec(labels=(), index_sets=((),)),
        predicate=True,
        update_program=nkdsl.identity().to_program(),
        amplitude=nkdsl.AmplitudeExpr.constant(1.0),
        emissions=(
            EmissionSpec(
                nkdsl.identity().to_program(),
                nkdsl.AmplitudeExpr.constant(1.0),
            ),
            EmissionSpec(
                nkdsl.scatter([0], [2]).to_program(),
                nkdsl.AmplitudeExpr.constant(2.0),
            ),
        ),
    )
    runner_global = jl.make_kbody_runner(term_global, hi.size, np.float64)
    xp_g, m_g, v_g = runner_global(jnp.asarray([0, 1, 2], dtype=jnp.int32))
    assert xp_g.shape == (2, 3)
    assert m_g.shape == (2,)
    assert v_g.shape == (2,)

    term_site = SymbolicIRTerm.create(
        name="s",
        iterator=KBodyIteratorSpec(labels=("i",), index_sets=((0,), (1,), (2,))),
        predicate=nkdsl.site("i") >= 0,
        update_program=nkdsl.shift("i", +1).to_program(),
        amplitude=nkdsl.AmplitudeExpr.constant(1.0),
    )
    runner_site = jl.make_kbody_runner(term_site, hi.size, np.float64)
    xp_s, m_s, v_s = runner_site(jnp.asarray([0, 1, 2], dtype=jnp.int32))
    assert xp_s.shape == (3, 3)
    assert m_s.shape == (3,)
    assert v_s.shape == (3,)

    prog = nkdsl.shift("i", +1).invalidate().to_program()
    env = {"site:i:index": jnp.int32(0), "site:i:value": jnp.int32(0)}
    x_prime, valid = jl.apply_update_program(jnp.asarray([0, 1], dtype=jnp.int32), prog, env, 2)
    assert bool(valid) is False
    np.testing.assert_array_equal(np.asarray(x_prime), np.asarray([1, 1]))


def test_build_compiled_operator_padding_branches():
    hi = nk.hilbert.Fock(n_max=2, N=2)

    def runner_short(x):
        return jnp.asarray([[x[0], x[1]]]), jnp.asarray([1.0]), jnp.asarray([True])

    op = jl.build_compiled_operator(
        hi,
        operator_name="pad",
        is_hermitian=True,
        output_dtype=np.float64,
        term_runners=[runner_short],
        total_padded_size=3,
    )

    x = jnp.asarray([0, 1], dtype=jnp.int32)
    xp, m = op.get_conn_padded(x)
    assert xp.shape == (3, 2)
    assert m.shape == (3,)


def test_build_compiled_operator_with_custom_connection_method():
    hi = nk.hilbert.Fock(n_max=2, N=2)

    class _ComputationalLikeOperator:
        def __init__(self, hilbert):
            self.hilbert = hilbert

        def get_conn_padded(self, x):
            return self._get_conn_padded(x)

    def runner_short(x):
        return jnp.asarray([[x[0], x[1]]]), jnp.asarray([2.0]), jnp.asarray([True])

    op = jl.build_compiled_operator(
        hi,
        operator_name="custom",
        is_hermitian=False,
        output_dtype=np.float32,
        term_runners=[runner_short],
        total_padded_size=1,
        operator_type=_ComputationalLikeOperator,
        connection_method="_get_conn_padded",
    )

    x = jnp.asarray([1, 0], dtype=jnp.int32)
    xp, m = op.get_conn_padded(x)
    np.testing.assert_array_equal(np.asarray(xp), np.asarray([[1, 0]], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(m), np.asarray([2.0], dtype=np.float32))
