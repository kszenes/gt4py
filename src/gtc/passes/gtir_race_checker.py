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

from dataclasses import dataclass, field
from typing import Set

import eve
from gtc import gtir


class _RaceChecker(eve.NodeVisitor):
    @dataclass
    class Context:
        stencil_name: str
        no_writes_after_reads: Set[str]

        writes: Set[str] = field(default_factory=set)
        reads: Set[str] = field(default_factory=set)

    def visit_Stencil(self, node: gtir.Stencil) -> None:
        api_reads_with_offset = (
            node.iter_tree()
            .if_isinstance(gtir.FieldAccess)
            .filter(lambda acc: acc.offset.i != 0 or acc.offset.j != 0)
            .getattr("name")
            .to_set()
        ) & {p.name for p in node.params}
        ctx = self.Context(stencil_name=node.name, no_writes_after_reads=api_reads_with_offset)
        self.visit(node.vertical_loops, ctx=ctx)

    def visit_ParAssignStmt(
        self,
        node: gtir.ParAssignStmt,
        *,
        ctx: Context,
    ) -> None:
        if node.left.name in ctx.reads and node.left.name in ctx.no_writes_after_reads:
            raise ValueError(
                f"{ctx.stencil_name}: Write after read race condition detected on {node.name}"
            )

        if isinstance(node.target, gtir.FieldAccess):
            ctx.writes.add(node.target.name)

    def visit_HorizontalRegion(self, node: gtir.HorizontalRegion, *, ctx: Context) -> None:
        self.visit(node.body, ctx=ctx)

        reads_with_offset = (
            node.body.iter_tree()
            .if_isinstance(gtir.FieldAccess)
            .filter(lambda acc: acc.offset.i != 0 or acc.offset.j != 0)
            .getattr("name")
            .to_set()
        )
        if any(name in ctx.writes for name in reads_with_offset):
            raise ValueError(
                f"{ctx.stencil_name}: Write before read with offset in horizontal region detected on {node.name}"
            )


def check_race_conditions(node: gtir.Stencil) -> gtir.Stencil:
    _RaceChecker().visit(node)
    return node
