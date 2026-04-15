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

import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

import nkdsl
from nkdsl.dsl import AbstractEmissionClause
from nkdsl.dsl import AbstractIteratorClause
from nkdsl.dsl import AbstractPredicateClause
from nkdsl.dsl import register
from nkdsl.dsl import register_emission_clause
from nkdsl.dsl import register_iterator_clause
from nkdsl.dsl import register_predicate_clause
from nkdsl.dsl.clauses import apply_emission_clause
from nkdsl.dsl.clauses import apply_iterator_clause
from nkdsl.dsl.clauses import apply_predicate_clause
from nkdsl.dsl.clauses import coerce_iterator_spec
from nkdsl.dsl.clauses import ensure_default_clause_registrations
from nkdsl.dsl.emissions.types import EmissionClauseSpec

pytestmark = pytest.mark.unit


def _small_hi(n_sites: int = 4):
    return nk.hilbert.Fock(n_max=5, N=n_sites)


def test_default_clause_names_are_exposed_in_registry_and_dir():
    hi = _small_hi()
    builder = nkdsl.SymbolicDiscreteJaxOperator(hi, "clauses")

    iterator_names = nkdsl.available_iterator_clause_names()
    predicate_names = nkdsl.available_predicate_clause_names()
    emission_names = nkdsl.available_emission_clause_names()

    assert "globally" in iterator_names
    assert "for_each_site" in iterator_names
    assert "for_each_pair" in iterator_names
    assert "for_each_distinct_pair" in iterator_names
    assert "for_each_triplet" in iterator_names
    assert "for_each_plaquette" in iterator_names
    assert "for_each" in iterator_names
    assert "where" in predicate_names
    assert "emit_if" in emission_names
    assert "emit_elseif" in emission_names
    assert "emit_else" in emission_names

    entries = dir(builder)
    assert "for_each_site" in entries
    assert "where" in entries
    assert "emit_if" in entries


def test_custom_iterator_clause_dynamic_method_executes():
    class EvenSitesIteratorClause(AbstractIteratorClause):
        clause_name = "even_sites_clause_test"

        def build_iterator(self, hilbert, label: str = "i"):
            n = int(hilbert.size)
            rows = tuple((k,) for k in range(n) if k % 2 == 0)
            return (str(label),), rows

    register_iterator_clause(EvenSitesIteratorClause, replace=True)

    hi = _small_hi(4)
    assert getattr(nkdsl.SymbolicDiscreteJaxOperator, "even_sites_clause_test", None) is None

    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "even-site")
        .even_sites_clause_test("i")
        .emit(nkdsl.identity(), matrix_element=nkdsl.site("i").value)
        .build()
        .compile()
    )

    x = jnp.asarray([3, 4, 5, 6], dtype=jnp.int32)
    xp, mels = op.get_conn_padded(x)

    np.testing.assert_array_equal(np.asarray(xp), np.asarray([[3, 4, 5, 6], [3, 4, 5, 6]]))
    np.testing.assert_allclose(np.asarray(mels), np.asarray([3.0, 5.0]))
    assert op.max_conn_size == 2


def test_custom_predicate_clause_composes_with_where():
    class AtLeastPredicateClause(AbstractPredicateClause):
        clause_name = "at_least_clause_test"

        def build_predicate(self, ctx, label: str = "i", cutoff: int = 0):
            return ctx.site(label).value >= cutoff

    register_predicate_clause(AtLeastPredicateClause, replace=True)

    hi = _small_hi(4)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "pred")
        .for_each_site("i")
        .at_least_clause_test("i", cutoff=1)
        .where(nkdsl.site("i") < 3)
        .emit(nkdsl.identity(), matrix_element=1.0)
        .build()
        .compile()
    )

    x = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
    _xp, mels = op.get_conn_padded(x)
    np.testing.assert_allclose(np.asarray(mels), np.asarray([0.0, 1.0, 1.0, 0.0]))


def test_register_decorator_accepts_iterator_predicate_and_emission_clause_classes():
    @register
    class DecoratorIteratorClause(AbstractIteratorClause):
        clause_name = "decorator_iterator_clause_test"

        def build_iterator(self, hilbert, label: str = "i"):
            return (str(label),), ((0,),)

    @register
    class DecoratorPredicateClause(AbstractPredicateClause):
        clause_name = "decorator_predicate_clause_test"

        def build_predicate(self, ctx, label: str = "i"):
            return ctx.eq(ctx.site(label).value, 1)

    @register
    class DecoratorEmissionClause(AbstractEmissionClause):
        clause_name = "decorator_emission_clause_test"

        def build_emission(self, ctx, label: str = "i"):
            del ctx
            return EmissionClauseSpec(
                mode="emit_if",
                predicate=nkdsl.site(label).value > 0,
                update=nkdsl.identity(),
                matrix_element=2.0,
            )

    hi = _small_hi(3)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "decorator")
        .decorator_iterator_clause_test("i")
        .decorator_predicate_clause_test("i")
        .decorator_emission_clause_test("i")
        .build()
        .compile()
    )

    _xp, mels = op.get_conn_padded(jnp.asarray([1, 0, 0], dtype=jnp.int32))
    np.testing.assert_allclose(np.asarray(mels), np.asarray([2.0]))


def test_registration_conflict_and_replace_behaviour():
    class ReplaceableIteratorClauseV1(AbstractIteratorClause):
        clause_name = "replace_iterator_clause_test"

        def build_iterator(self, hilbert, label: str = "i"):
            return (str(label),), ((0,),)

    class ReplaceableIteratorClauseV2(AbstractIteratorClause):
        clause_name = "replace_iterator_clause_test"

        def build_iterator(self, hilbert, label: str = "i"):
            n = int(hilbert.size)
            return (str(label),), tuple((k,) for k in range(n))

    register_iterator_clause(ReplaceableIteratorClauseV1, replace=True)
    with pytest.raises(ValueError, match="already registered"):
        register_iterator_clause(ReplaceableIteratorClauseV2)

    register_iterator_clause(ReplaceableIteratorClauseV2, replace=True)

    hi = _small_hi(3)
    op = (
        nkdsl.SymbolicDiscreteJaxOperator(hi, "replace")
        .replace_iterator_clause_test("i")
        .emit(nkdsl.identity(), matrix_element=1.0)
        .build()
        .compile()
    )
    assert op.max_conn_size == 3


def test_reserved_or_invalid_clause_names_are_rejected():
    class AnyIteratorClause(AbstractIteratorClause):
        def build_iterator(self, hilbert):
            return (), ((),)

    with pytest.raises(ValueError, match="reserved"):
        register_iterator_clause(AnyIteratorClause, name="build", replace=True)

    class AnyPredicateClause(AbstractPredicateClause):
        def build_predicate(self, ctx):
            return True

    with pytest.raises(ValueError, match="valid Python identifier"):
        register_predicate_clause(AnyPredicateClause, name="bad-name!", replace=True)

    with pytest.raises(ValueError, match="starting with '_'"):
        register_predicate_clause(AnyPredicateClause, name="_hidden", replace=True)


def test_register_rejects_non_clause_classes():
    class Plain:
        pass

    with pytest.raises(TypeError, match="expects a subclass"):
        register(Plain)

    with pytest.raises(TypeError, match="Iterator clause must inherit"):
        register_iterator_clause(Plain)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Predicate clause must inherit"):
        register_predicate_clause(Plain)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Emission clause must inherit"):
        register_emission_clause(Plain)  # type: ignore[arg-type]


def test_iterator_clause_return_validation_errors():
    class InvalidTypeIteratorClause(AbstractIteratorClause):
        clause_name = "invalid_type_iterator_clause_test"

        def build_iterator(self, hilbert):
            return 5

    register_iterator_clause(InvalidTypeIteratorClause, replace=True)
    with pytest.raises(TypeError, match="must return KBodyIteratorSpec"):
        nkdsl.SymbolicDiscreteJaxOperator(_small_hi(), "x").invalid_type_iterator_clause_test()

    class EmptyRowsIteratorClause(AbstractIteratorClause):
        clause_name = "empty_rows_iterator_clause_test"

        def build_iterator(self, hilbert):
            return ("i",), ()

    register_iterator_clause(EmptyRowsIteratorClause, replace=True)
    with pytest.raises(ValueError, match="at least one index tuple"):
        nkdsl.SymbolicDiscreteJaxOperator(_small_hi(), "x").empty_rows_iterator_clause_test()

    class WrongArityIteratorClause(AbstractIteratorClause):
        clause_name = "wrong_arity_iterator_clause_test"

        def build_iterator(self, hilbert):
            return ("i", "j"), ((0,),)

    register_iterator_clause(WrongArityIteratorClause, replace=True)
    with pytest.raises(ValueError, match="labels of length 2"):
        nkdsl.SymbolicDiscreteJaxOperator(_small_hi(), "x").wrong_arity_iterator_clause_test()


def test_abstract_clause_bases_are_not_instantiable():
    hi = _small_hi()
    builder = nkdsl.SymbolicDiscreteJaxOperator(hi, "abstract")

    with pytest.raises(TypeError):
        AbstractIteratorClause(builder)

    with pytest.raises(TypeError):
        AbstractPredicateClause(builder)

    with pytest.raises(TypeError):
        AbstractEmissionClause(builder)


def test_register_factory_forms_and_empty_name_guard():
    class FactoryIteratorClause(AbstractIteratorClause):
        clause_name = "factory_iterator_clause_test"

        def build_iterator(self, hilbert):
            return (), ((),)

    class FactoryPredicateClause(AbstractPredicateClause):
        clause_name = "factory_predicate_clause_test"

        def build_predicate(self, ctx):
            return True

    class FactoryEmissionClause(AbstractEmissionClause):
        clause_name = "factory_emission_clause_test"

        def build_emission(self, ctx):
            del ctx
            return EmissionClauseSpec(mode="emit", update=nkdsl.identity(), matrix_element=1.0)

    iterator_decorator = register_iterator_clause(name="factory_iter_clause_test", replace=True)
    registered_iter = iterator_decorator(FactoryIteratorClause)
    assert registered_iter is FactoryIteratorClause

    predicate_decorator = register_predicate_clause(name="factory_pred_clause_test", replace=True)
    registered_pred = predicate_decorator(FactoryPredicateClause)
    assert registered_pred is FactoryPredicateClause

    emission_decorator = register_emission_clause(name="factory_em_clause_test", replace=True)
    registered_em = emission_decorator(FactoryEmissionClause)
    assert registered_em is FactoryEmissionClause

    generic_decorator = register(name="factory_generic_clause_test", replace=True)
    generic_registered = generic_decorator(FactoryPredicateClause)
    assert generic_registered is FactoryPredicateClause

    with pytest.raises(ValueError, match="non-empty string"):
        register_iterator_clause(name="   ")(FactoryIteratorClause)


def test_coerce_iterator_spec_accepts_object_with_labels_and_index_sets():
    class ObjSpec:
        labels = ("i",)
        index_sets = ((1,),)

    coerced = coerce_iterator_spec(ObjSpec())
    assert coerced.labels == ("i",)
    assert coerced.index_sets == ((1,),)


def test_predicate_conflict_unknown_apply_and_globals_guard_paths():
    class ConflictPredicateV1(AbstractPredicateClause):
        clause_name = "predicate_conflict_clause_test"

        def build_predicate(self, ctx):
            return True

    class ConflictPredicateV2(AbstractPredicateClause):
        clause_name = "predicate_conflict_clause_test"

        def build_predicate(self, ctx):
            return False

    register_predicate_clause(ConflictPredicateV1, replace=True)
    with pytest.raises(ValueError, match="already registered"):
        register_predicate_clause(ConflictPredicateV2)

    hi = _small_hi()
    builder = nkdsl.SymbolicDiscreteJaxOperator(hi, "unknowns")

    with pytest.raises(AttributeError, match="Unknown iterator clause"):
        apply_iterator_clause(builder, "does_not_exist_iterator_clause")

    with pytest.raises(AttributeError, match="Unknown predicate clause"):
        apply_predicate_clause(builder, "does_not_exist_predicate_clause")

    with pytest.raises(AttributeError, match="Unknown emission clause"):
        apply_emission_clause(builder, "does_not_exist_emission_clause")

    with pytest.raises(TypeError, match="does not accept positional"):
        apply_iterator_clause(builder, "globally", 1)


def test_default_clause_registration_is_idempotent():
    before_iter = set(nkdsl.available_iterator_clause_names())
    before_pred = set(nkdsl.available_predicate_clause_names())
    before_em = set(nkdsl.available_emission_clause_names())

    ensure_default_clause_registrations()
    ensure_default_clause_registrations()

    after_iter = set(nkdsl.available_iterator_clause_names())
    after_pred = set(nkdsl.available_predicate_clause_names())
    after_em = set(nkdsl.available_emission_clause_names())

    assert before_iter <= after_iter
    assert before_pred <= after_pred
    assert before_em <= after_em
