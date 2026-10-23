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


"""Duplicate extension registrations must preserve the running DSL semantics."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest

import nkdsl

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("family", ["iterator", "predicate", "emission"])
@pytest.mark.parametrize("style", ["direct", "decorator", "generic", "generic_decorator"])
def test_duplicate_keeps_original_runtime_behavior_until_explicit_replacement(
    family, style, monkeypatch
):
    module = importlib.import_module(f"nkdsl.dsl.{family}s.registry")
    mapping = f"_{family.upper()}_CLAUSES"
    monkeypatch.setattr(module, mapping, getattr(module, mapping).copy())
    name = f"duplicate_{family}_regression"
    if family == "iterator":
        base = nkdsl.AbstractIteratorClause

        def original_method(self, hilbert, label="i"):
            return (label,), ((0,),)

        def replacement_method(self, hilbert, label="i"):
            return (label,), tuple((i,) for i in range(hilbert.size))

        method_name, original_value, replacement_value = "build_iterator", 1, 3
    elif family == "predicate":
        base = nkdsl.AbstractPredicateClause

        def original_method(self, ctx):
            return True

        def replacement_method(self, ctx):
            return False

        method_name, original_value, replacement_value = "build_predicate", 3, 0
    else:
        base = nkdsl.AbstractEmissionClause

        def original_method(self, ctx):
            return nkdsl.EmissionClauseSpec(mode="emit", update=nkdsl.identity(), matrix_element=2)

        def replacement_method(self, ctx):
            return nkdsl.EmissionClauseSpec(mode="emit", update=nkdsl.identity(), matrix_element=7)

        method_name, original_value, replacement_value = "build_emission", 6, 21

    original = type(
        "Original", (base,), {method_name: original_method, "clause_name": "original_native_name"}
    )
    replacement = type(
        "Replacement",
        (base,),
        {method_name: replacement_method, "clause_name": "replacement_native_name"},
    )
    specific = getattr(nkdsl, f"register_{family}_clause")
    register = nkdsl.register if style.startswith("generic") else specific

    def install(cls, **kwargs):
        if style.endswith("decorator"):
            return register(name=name, **kwargs)(cls)
        return register(cls, name=name, **kwargs)

    def evaluate():
        builder = nkdsl.SymbolicDiscreteJaxOperator(nk.hilbert.Fock(n_max=1, N=3), "duplicate")
        if family == "iterator":
            builder = getattr(builder, name)("i").emit(nkdsl.identity(), matrix_element=1)
        else:
            builder = builder.for_each_site("i")
            builder = getattr(builder, name)()
            if family == "predicate":
                builder = builder.emit(nkdsl.identity(), matrix_element=1)
        compiled = builder.compile(cache=False)
        _, mels = jax.jit(compiled.get_conn_padded)(jnp.asarray([0, 1, 0]))
        return np.asarray(mels).sum()

    assert install(original) is original
    assert evaluate() == original_value
    for _ in range(2):
        with pytest.warns(nkdsl.RegistrationWarning, match="Keeping the existing.*replace=True"):
            assert install(replacement) is replacement
        assert replacement.clause_name == "replacement_native_name"
        assert getattr(module, f"resolve_{family}_clause")(name) is original
        assert evaluate() == original_value

    assert install(replacement, replace=True) is replacement
    assert replacement.clause_name == name
    assert getattr(module, f"resolve_{family}_clause")(name) is replacement
    assert evaluate() == replacement_value


@pytest.mark.parametrize("family", ["iterator", "predicate", "emission"])
def test_reregistering_same_class_warns_without_changing_the_registry(family, monkeypatch):
    module = importlib.import_module(f"nkdsl.dsl.{family}s.registry")
    mapping = f"_{family.upper()}_CLAUSES"
    monkeypatch.setattr(module, mapping, getattr(module, mapping).copy())
    register = getattr(nkdsl, f"register_{family}_clause")
    name = f"same_{family}_regression"
    base = getattr(nkdsl, f"Abstract{family.capitalize()}Clause")
    cls = type("SameClause", (base,), {"clause_name": name})
    register(cls, name=name)
    with pytest.warns(nkdsl.RegistrationWarning, match="replace=True"):
        assert register(cls, name=name) is cls
    assert getattr(module, f"resolve_{family}_clause")(name) is cls
