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

import itertools
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from eve.traits import SymbolTableTrait
from eve.visitors import NodeTranslator

from .. import common, oir
from . import jaxir


class OirToJaxir(NodeTranslator):
    """Lower from optimizable IR (OIR) to JAX IR (JaxIR)."""

    @dataclass
    class ComputationContext:
        """Top Level Context."""

        temp_defs: typing.OrderedDict[str, jaxir.VectorAssign] = field(
            default_factory=lambda: OrderedDict({})
        )

        mask_temp_counter: int = 0

        def ensure_temp_defined(self, temp: Union[oir.FieldAccess, jaxir.FieldSlice]) -> None:
            if temp.name not in self.temp_defs:
                self.temp_defs[str(temp.name)] = jaxir.VectorAssign(
                    left=jaxir.VectorTemp(name=str(temp.name), dtype=temp.dtype),
                    right=jaxir.EmptyTemp(dtype=temp.dtype),
                )

    contexts = (SymbolTableTrait.symtable_merger,)

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> jaxir.Computation:
        ctx = self.ComputationContext()
        vertical_passes = list(
            itertools.chain(
                *[self.visit(vloop, ctx=ctx, **kwargs) for vloop in node.vertical_loops]
            )
        )
        field_names: List[str] = []
        scalar_names: List[str] = []
        field_decls: List[jaxir.FieldDecl] = []
        for decl in node.params:
            if isinstance(decl, oir.FieldDecl):
                field_names.append(str(decl.name))
                field_decls.append(self.visit(decl))
            else:
                scalar_names.append(decl.name)
        return jaxir.Computation(
            field_decls=field_decls,
            field_params=field_names,
            params=[decl.name for decl in node.params],
            vertical_passes=vertical_passes,
        )

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs) -> List[jaxir.VerticalPass]:
        return self.visit(node.sections, v_caches=node.caches, loop_order=node.loop_order, **kwargs)

    def visit_VerticalLoopSection(
        self,
        node: oir.VerticalLoopSection,
        *,
        loop_order: common.LoopOrder,
        ctx: Optional[ComputationContext] = None,
        v_caches: List[oir.CacheDesc] = None,
        **kwargs: Any,
    ) -> jaxir.VerticalPass:
        ctx = ctx or self.ComputationContext()
        defined_temps = set(ctx.temp_defs.keys())
        kwargs.update(
            {
                "parallel_k": True if loop_order == common.LoopOrder.PARALLEL else False,
                "lower_k": node.interval.start,
                "upper_k": node.interval.end,
            }
        )
        body = self.visit(node.horizontal_executions, ctx=ctx, **kwargs)
        undef_temps = [
            temp_def for name, temp_def in ctx.temp_defs.items() if name not in defined_temps
        ]
        return jaxir.VerticalPass(
            body=body,
            temp_defs=undef_temps,
            lower=self.visit(node.interval.start, ctx=ctx, **kwargs),
            upper=self.visit(node.interval.end, ctx=ctx, **kwargs),
            direction=self.visit(loop_order, ctx=ctx, **kwargs),
        )

    def visit_HorizontalExecution(
        self,
        node: oir.HorizontalExecution,
        *,
        ctx: Optional[ComputationContext] = None,
        **kwargs: Any,
    ) -> jaxir.HorizontalBlock:
        return jaxir.HorizontalBlock(
            body=self.visit(node.body, ctx=ctx, **kwargs),
        )

    def visit_MaskStmt(
        self,
        node: oir.MaskStmt,
        *,
        ctx: ComputationContext,
        parallel_k: bool,
        **kwargs,
    ) -> jaxir.MaskBlock:
        mask_expr = self.visit(node.mask, ctx=ctx, parallel_k=parallel_k, broadcast=True, **kwargs)
        if isinstance(mask_expr, jaxir.FieldSlice):
            mask_name = mask_expr.name
            mask = mask_expr
        else:
            mask_name = f"_mask_{ctx.mask_temp_counter}"
            mask = jaxir.VectorTemp(
                name=mask_name,
            )
            ctx.mask_temp_counter += 1

        return jaxir.MaskBlock(
            mask=mask_expr,
            mask_name=mask_name,
            body=self.visit(node.body, ctx=ctx, parallel_k=parallel_k, mask=mask, **kwargs),
        )

    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        *,
        ctx: Optional[ComputationContext] = None,
        mask: Optional[jaxir.VectorExpression] = None,
        **kwargs: Any,
    ) -> jaxir.VectorAssign:
        ctx = ctx or self.ComputationContext()
        if isinstance(
            kwargs["symtable"].get(node.left.name, None), (oir.Temporary, oir.LocalScalar)
        ):
            ctx.ensure_temp_defined(node.left)
        return jaxir.VectorAssign(
            left=self.visit(node.left, ctx=ctx, is_lvalue=True, **kwargs),
            right=self.visit(node.right, ctx=ctx, broadcast=True, **kwargs),
            mask=mask,
        )

    def visit_Cast(
        self,
        node: oir.Cast,
        *,
        ctx: Optional[ComputationContext] = None,
        broadcast: bool = False,
        **kwargs: Any,
    ) -> Union[jaxir.Cast, jaxir.BroadCast]:
        cast = jaxir.Cast(
            dtype=self.visit(node.dtype, ctx=ctx, **kwargs),
            expr=self.visit(node.expr, ctx=ctx, broadcast=False, **kwargs),
        )
        if broadcast:
            return jaxir.BroadCast(expr=cast, dtype=node.dtype)
        return cast

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        ctx: ComputationContext,
        parallel_k: bool,
        **kwargs: Any,
    ) -> jaxir.FieldSlice:
        dims = (
            decl.dimensions if (decl := kwargs["symtable"].get(node.name)) else (True, True, True)
        )
        if isinstance(node.offset, common.CartesianOffset):
            i_offset = jaxir.AxisOffset.i(node.offset.i) if dims[0] else None
            j_offset = jaxir.AxisOffset.j(node.offset.j) if dims[1] else None
            k_offset = jaxir.AxisOffset.k(node.offset.k, parallel=parallel_k) if dims[2] else None
        else:
            i_offset = jaxir.AxisOffset.i(0)
            j_offset = jaxir.AxisOffset.j(0)
            k_offset = jaxir.VariableKOffset(
                k=self.visit(node.offset.k, ctx=ctx, parallel_k=parallel_k, **kwargs)
            )
        return jaxir.FieldSlice(
            name=str(node.name),
            i_offset=i_offset,
            j_offset=j_offset,
            k_offset=k_offset,
            data_index=self.visit(node.data_index, ctx=ctx, parallel_k=parallel_k, **kwargs),
        )

    def visit_FieldDecl(self, node: oir.FieldDecl, **kwargs: Any) -> jaxir.FieldDecl:
        return jaxir.FieldDecl(
            name=node.name,
            dtype=self.visit(node.dtype),
            dimensions=node.dimensions,
            data_dims=node.data_dims,
        )

    def visit_BinaryOp(
        self, node: oir.BinaryOp, *, ctx: Optional[ComputationContext] = None, **kwargs: Any
    ) -> Union[jaxir.VectorArithmetic, jaxir.VectorLogic]:
        kwargs["broadcast"] = True
        left = self.visit(node.left, ctx=ctx, **kwargs)
        right = self.visit(node.right, ctx=ctx, **kwargs)

        if isinstance(node.op, common.LogicalOperator):
            return jaxir.VectorLogic(op=node.op, left=left, right=right)

        return jaxir.VectorArithmetic(
            op=node.op,
            left=left,
            right=right,
        )

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> jaxir.VectorUnaryOp:
        kwargs["broadcast"] = True
        return jaxir.VectorUnaryOp(op=node.op, expr=self.visit(node.expr, **kwargs))

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> jaxir.VectorTernaryOp:
        kwargs["broadcast"] = True

        return jaxir.VectorTernaryOp(
            cond=self.visit(node.cond, **kwargs),
            true_expr=self.visit(node.true_expr, **kwargs),
            false_expr=self.visit(node.false_expr, **kwargs),
        )

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> jaxir.NativeFuncCall:
        kwargs["broadcast"] = True
        return jaxir.NativeFuncCall(
            func=self.visit(node.func, **kwargs),
            args=self.visit(node.args, **kwargs),
        )

    def visit_Literal(
        self, node: oir.Literal, *, broadcast: bool = False, **kwargs: Any
    ) -> Union[jaxir.Literal, jaxir.BroadCast]:
        literal = jaxir.Literal(value=self.visit(node.value, **kwargs), dtype=node.dtype)
        if broadcast:
            return jaxir.BroadCast(expr=literal, dtype=node.dtype)
        return literal

    def visit_ScalarAccess(
        self,
        node: oir.ScalarAccess,
        *,
        broadcast: bool = False,
        ctx: Optional[ComputationContext] = None,
        **kwargs: Any,
    ) -> Union[jaxir.BroadCast, jaxir.NamedScalar]:
        ctx = ctx or self.ComputationContext()
        if node.name in ctx.temp_defs:
            name = jaxir.VectorTemp(name=self.visit(node.name, **kwargs))
        else:
            name = jaxir.NamedScalar(
                name=self.visit(node.name, **kwargs), dtype=self.visit(node.dtype, **kwargs)
            )
        if broadcast:
            return jaxir.BroadCast(expr=name, dtype=name.dtype)
        return name
