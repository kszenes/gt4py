# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from pydantic.error_wrappers import ValidationError

from eve import SourceLocation
from gtc.common import ArithmeticOperator, DataType, LevelMarker, LoopOrder
from gtc.gtir import (
    AxisBound,
    CartesianOffset,
    Decl,
    Expr,
    FieldAccess,
    FieldDecl,
    Interval,
    ParAssignStmt,
    Stencil,
    Stmt,
    VerticalLoop,
)

from .gtir_utils import (
    DummyExpr,
    FieldAccessBuilder,
    ParAssignStmtBuilder,
    StencilBuilder,
    VerticalLoopBuilder,
)


ARITHMETIC_TYPE = DataType.FLOAT32
ANOTHER_ARITHMETIC_TYPE = DataType.INT32
A_ARITHMETIC_OPERATOR = ArithmeticOperator.ADD


@pytest.fixture
def copy_assign():
    yield ParAssignStmt(
        loc=SourceLocation(line=3, column=2, source="copy_gtir"),
        left=FieldAccess.centered(
            name="a", loc=SourceLocation(line=3, column=1, source="copy_gtir")
        ),
        right=FieldAccess.centered(
            name="b", loc=SourceLocation(line=3, column=3, source="copy_gtir")
        ),
    )


@pytest.fixture
def interval(copy_assign):
    yield Interval(
        loc=SourceLocation(line=2, column=11, source="copy_gtir"),
        start=AxisBound(level=LevelMarker.START, offset=0),
        end=AxisBound(level=LevelMarker.END, offset=0),
    )


@pytest.fixture
def copy_v_loop(copy_assign, interval):
    yield VerticalLoop(
        loc=SourceLocation(line=2, column=1, source="copy_gtir"),
        loop_order=LoopOrder.FORWARD,
        interval=interval,
        body=[copy_assign],
        temporaries=[],
    )


@pytest.fixture
def copy_computation(copy_v_loop):
    yield Stencil(
        name="copy_gtir",
        loc=SourceLocation(line=1, column=1, source="copy_gtir"),
        params=[
            FieldDecl(name="a", dtype=DataType.FLOAT32, dimensions=(True, True, True)),
            FieldDecl(name="b", dtype=DataType.FLOAT32, dimensions=(True, True, True)),
        ],
        vertical_loops=[copy_v_loop],
    )


def test_copy(copy_computation):
    assert copy_computation
    assert copy_computation.param_names == ["a", "b"]


@pytest.mark.parametrize(
    "invalid_node",
    [Decl, Expr, Stmt],
)
def test_abstract_classes_not_instantiatable(invalid_node):
    with pytest.raises(TypeError):
        invalid_node()


def test_can_have_vertical_offset():
    ParAssignStmt(
        left=FieldAccessBuilder("foo").offset(CartesianOffset(i=0, j=0, k=1)).build(),
        right=DummyExpr(),
    )


@pytest.mark.parametrize(
    "assign_stmt_with_offset",
    [
        lambda: ParAssignStmt(
            left=FieldAccessBuilder("foo").offset(CartesianOffset(i=1, j=0, k=0)).build(),
            right=DummyExpr(),
        ),
        lambda: ParAssignStmt(
            left=FieldAccessBuilder("foo").offset(CartesianOffset(i=0, j=1, k=0)).build(),
            right=DummyExpr(),
        ),
    ],
)
def test_no_horizontal_offset_allowed(assign_stmt_with_offset):
    with pytest.raises(ValidationError, match=r"must not have .*horizontal offset"):
        assign_stmt_with_offset()


def test_symbolref_without_decl():
    with pytest.raises(ValidationError, match=r"Symbols.*not found"):
        StencilBuilder().add_par_assign_stmt(
            ParAssignStmtBuilder("out_field", "in_field").build()
        ).build()


def test_assign_to_ik_fwd():
    out_name = "ik_field"
    in_name = "other_ik_field"
    with pytest.raises(ValidationError, match=r"Not allowed to assign to ik-field"):
        (
            StencilBuilder(name="assign_to_ik_fwd")
            .add_param(
                FieldDecl(name=out_name, dtype=DataType.FLOAT32, dimensions=(True, False, True)),
            )
            .add_param(
                FieldDecl(name=in_name, dtype=DataType.FLOAT32, dimensions=(True, False, True)),
            )
            .add_vertical_loop(
                VerticalLoopBuilder()
                .set_loop_order(LoopOrder.FORWARD)
                .add_stmt(ParAssignStmtBuilder(left_name=out_name, right_name=in_name).build())
                .build()
            )
            .build()
        )


def test_assign_to_ij_par():
    out_name = "ij_field"
    in_name = "other_field"
    with pytest.raises(
        ValidationError, match=r"Not allowed to assign to ij-field `ij_field` in PARALLEL"
    ):
        (
            StencilBuilder(name="assign_to_ij_par")
            .add_param(
                FieldDecl(name=out_name, dtype=DataType.FLOAT32, dimensions=(True, True, False)),
            )
            .add_param(
                FieldDecl(name=in_name, dtype=DataType.FLOAT32, dimensions=(True, True, True)),
            )
            .add_vertical_loop(
                VerticalLoopBuilder()
                .set_loop_order(LoopOrder.PARALLEL)
                .add_stmt(ParAssignStmtBuilder(left_name=out_name, right_name=in_name).build())
                .build()
            )
            .build()
        )
