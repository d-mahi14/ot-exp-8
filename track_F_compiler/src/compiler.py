"""Optimization Pipeline and IR Builder for Track F Mini-Compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

from track_F_compiler.src.interpreter import ExecutionResult, IRInterpreter
from track_F_compiler.src.ir import Instruction, OpCode
from track_F_compiler.src.optimizations import (
    RegisterAllocationResult,
    run_common_subexpression_elimination,
    run_constant_folding_and_propagation,
    run_dead_code_elimination,
    run_loop_invariant_code_motion,
    solve_ilp_register_allocation,
)


class IRBuilder:
    """Convenience fluent builder for creating Three-Address Code programs."""

    def __init__(self) -> None:
        self.instructions: List[Instruction] = []

    def label(self, name: str) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.LABEL, label=name))
        return self

    def const(self, dest: str, val: Union[int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.CONST, dest=dest, src1=val))
        return self

    def copy(self, dest: str, src: str) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.COPY, dest=dest, src1=src))
        return self

    def add(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.ADD, dest=dest, src1=src1, src2=src2))
        return self

    def sub(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.SUB, dest=dest, src1=src1, src2=src2))
        return self

    def mul(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.MUL, dest=dest, src1=src1, src2=src2))
        return self

    def div(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.DIV, dest=dest, src1=src1, src2=src2))
        return self

    def mod(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.MOD, dest=dest, src1=src1, src2=src2))
        return self

    def lt(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.LT, dest=dest, src1=src1, src2=src2))
        return self

    def le(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.LE, dest=dest, src1=src1, src2=src2))
        return self

    def gt(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.GT, dest=dest, src1=src1, src2=src2))
        return self

    def ge(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.GE, dest=dest, src1=src1, src2=src2))
        return self

    def eq(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.EQ, dest=dest, src1=src1, src2=src2))
        return self

    def ne(self, dest: str, src1: Union[str, int, float], src2: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.NE, dest=dest, src1=src1, src2=src2))
        return self

    def jmp(self, label: str) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.JMP, label=label))
        return self

    def jmp_if(self, cond: str, label: str) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.JMP_IF, src1=cond, label=label))
        return self

    def jmp_if_false(self, cond: str, label: str) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.JMP_IF_FALSE, src1=cond, label=label))
        return self

    def ret(self, val: Union[str, int, float]) -> IRBuilder:
        self.instructions.append(Instruction(op=OpCode.RET, src1=val))
        return self

    def build(self) -> List[Instruction]:
        return list(self.instructions)


@dataclass
class CompilationResult:
    original_instructions: List[Instruction]
    optimized_instructions: List[Instruction]
    static_instruction_reduction: int
    static_reduction_pct: float
    pass_history: Dict[str, int]
    register_allocation: Optional[RegisterAllocationResult] = None


class MiniCompiler:
    """Orchestrates optimization pipeline passes on TAC instruction streams."""

    def __init__(self, register_count: int = 4) -> None:
        self.register_count = register_count

    def optimize(
        self,
        instructions: List[Instruction],
        passes: Sequence[str] = ("const_fold", "cse", "licm", "dce"),
        apply_reg_alloc: bool = False,
    ) -> CompilationResult:
        """Run the selected optimization passes sequentially to fixed-point convergence."""
        curr = list(instructions)
        pass_history: Dict[str, int] = {"initial": len(curr)}

        # Run multi-pass optimization
        for p in passes:
            if p == "const_fold":
                curr = run_constant_folding_and_propagation(curr)
                pass_history["const_fold"] = len(curr)
            elif p == "cse":
                curr = run_common_subexpression_elimination(curr)
                pass_history["cse"] = len(curr)
            elif p == "licm":
                curr = run_loop_invariant_code_motion(curr)
                pass_history["licm"] = len(curr)
            elif p == "dce":
                curr = run_dead_code_elimination(curr)
                pass_history["dce"] = len(curr)

        # Final cleanup DCE pass to prune dead temps created by CSE/folding
        curr = run_dead_code_elimination(curr)
        pass_history["final_dce"] = len(curr)

        reg_res = None
        if apply_reg_alloc:
            reg_res = solve_ilp_register_allocation(curr, num_registers=self.register_count)
            curr = reg_res.rewritten_instructions
            pass_history["after_reg_alloc"] = len(curr)

        orig_len = len(instructions)
        opt_len = len(curr)
        reduction = orig_len - opt_len
        pct = (reduction / orig_len * 100.0) if orig_len > 0 else 0.0

        return CompilationResult(
            original_instructions=instructions,
            optimized_instructions=curr,
            static_instruction_reduction=reduction,
            static_reduction_pct=pct,
            pass_history=pass_history,
            register_allocation=reg_res,
        )
