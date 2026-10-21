"""Deterministic Virtual Machine / Interpreter for Three-Address Code (TAC) IR.

Executes TAC instructions, tracks dynamic execution statistics (instruction count, simulated cycles),
and produces output states for mathematical equivalence verification.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from track_F_compiler.src.ir import Instruction, OpCode


@dataclass
class ExecutionResult:
    return_value: Optional[Union[int, float]]
    variables: Dict[str, Union[int, float]]
    dynamic_instruction_count: int
    simulated_cycles: int
    execution_time_sec: float
    terminated_normally: bool


class IRInterpreter:
    """Virtual machine executing TAC instruction streams."""

    # Simulated microarchitectural cycle costs
    CYCLE_COSTS = {
        OpCode.CONST: 1,
        OpCode.COPY: 1,
        OpCode.ADD: 1,
        OpCode.SUB: 1,
        OpCode.MUL: 2,
        OpCode.DIV: 4,
        OpCode.MOD: 4,
        OpCode.LT: 1,
        OpCode.LE: 1,
        OpCode.GT: 1,
        OpCode.GE: 1,
        OpCode.EQ: 1,
        OpCode.NE: 1,
        OpCode.JMP: 1,
        OpCode.JMP_IF: 2,
        OpCode.JMP_IF_FALSE: 2,
        OpCode.LABEL: 0,
        OpCode.RET: 1,
        OpCode.SPILL_STORE: 3,  # Memory write latency
        OpCode.SPILL_LOAD: 3,   # Memory read latency
    }

    def __init__(self, max_steps: int = 500_000) -> None:
        self.max_steps = max_steps

    def execute(
        self,
        instructions: List[Instruction],
        initial_vars: Optional[Dict[str, Union[int, float]]] = None,
    ) -> ExecutionResult:
        """Execute a list of TAC instructions to completion or return."""
        t0 = time.perf_counter()
        env: Dict[str, Union[int, float]] = dict(initial_vars or {})
        spill_memory: Dict[str, Union[int, float]] = {}

        # Build label index map
        label_map: Dict[str, int] = {}
        for idx, instr in enumerate(instructions):
            if instr.op == OpCode.LABEL and instr.label:
                label_map[instr.label] = idx

        pc = 0
        step_count = 0
        total_cycles = 0
        ret_val = None
        n = len(instructions)

        def eval_operand(operand: Optional[Union[str, int, float]]) -> Union[int, float]:
            if operand is None:
                return 0
            if isinstance(operand, (int, float)):
                return operand
            if operand in env:
                return env[operand]
            # Try parsing numeric string
            try:
                if "." in operand:
                    return float(operand)
                return int(operand)
            except ValueError:
                # Default uninitialized variable to 0
                return 0

        while 0 <= pc < n and step_count < self.max_steps:
            instr = instructions[pc]
            step_count += 1
            total_cycles += self.CYCLE_COSTS.get(instr.op, 1)

            if instr.op == OpCode.LABEL:
                pc += 1

            elif instr.op == OpCode.CONST:
                assert instr.dest is not None
                env[instr.dest] = eval_operand(instr.src1)
                pc += 1

            elif instr.op == OpCode.COPY:
                assert instr.dest is not None
                env[instr.dest] = eval_operand(instr.src1)
                pc += 1

            elif instr.op == OpCode.ADD:
                assert instr.dest is not None
                env[instr.dest] = eval_operand(instr.src1) + eval_operand(instr.src2)
                pc += 1

            elif instr.op == OpCode.SUB:
                assert instr.dest is not None
                env[instr.dest] = eval_operand(instr.src1) - eval_operand(instr.src2)
                pc += 1

            elif instr.op == OpCode.MUL:
                assert instr.dest is not None
                env[instr.dest] = eval_operand(instr.src1) * eval_operand(instr.src2)
                pc += 1

            elif instr.op == OpCode.DIV:
                assert instr.dest is not None
                denom = eval_operand(instr.src2)
                if denom == 0:
                    raise ZeroDivisionError("Division by zero in TAC execution")
                # Integer division if both are ints
                v1 = eval_operand(instr.src1)
                if isinstance(v1, int) and isinstance(denom, int):
                    env[instr.dest] = v1 // denom
                else:
                    env[instr.dest] = v1 / denom
                pc += 1

            elif instr.op == OpCode.MOD:
                assert instr.dest is not None
                denom = eval_operand(instr.src2)
                if denom == 0:
                    raise ZeroDivisionError("Modulo by zero in TAC execution")
                env[instr.dest] = eval_operand(instr.src1) % denom
                pc += 1

            elif instr.op == OpCode.LT:
                assert instr.dest is not None
                env[instr.dest] = int(eval_operand(instr.src1) < eval_operand(instr.src2))
                pc += 1

            elif instr.op == OpCode.LE:
                assert instr.dest is not None
                env[instr.dest] = int(eval_operand(instr.src1) <= eval_operand(instr.src2))
                pc += 1

            elif instr.op == OpCode.GT:
                assert instr.dest is not None
                env[instr.dest] = int(eval_operand(instr.src1) > eval_operand(instr.src2))
                pc += 1

            elif instr.op == OpCode.GE:
                assert instr.dest is not None
                env[instr.dest] = int(eval_operand(instr.src1) >= eval_operand(instr.src2))
                pc += 1

            elif instr.op == OpCode.EQ:
                assert instr.dest is not None
                env[instr.dest] = int(eval_operand(instr.src1) == eval_operand(instr.src2))
                pc += 1

            elif instr.op == OpCode.NE:
                assert instr.dest is not None
                env[instr.dest] = int(eval_operand(instr.src1) != eval_operand(instr.src2))
                pc += 1

            elif instr.op == OpCode.JMP:
                assert instr.label is not None
                pc = label_map[instr.label]

            elif instr.op == OpCode.JMP_IF:
                assert instr.label is not None
                cond_val = eval_operand(instr.src1)
                if cond_val != 0:
                    pc = label_map[instr.label]
                else:
                    pc += 1

            elif instr.op == OpCode.JMP_IF_FALSE:
                assert instr.label is not None
                cond_val = eval_operand(instr.src1)
                if cond_val == 0:
                    pc = label_map[instr.label]
                else:
                    pc += 1

            elif instr.op == OpCode.RET:
                ret_val = eval_operand(instr.src1)
                break

            elif instr.op == OpCode.SPILL_STORE:
                assert instr.dest is not None
                spill_memory[instr.dest] = eval_operand(instr.src1)
                pc += 1

            elif instr.op == OpCode.SPILL_LOAD:
                assert instr.dest is not None
                assert instr.src1 is not None
                env[instr.dest] = spill_memory.get(str(instr.src1), 0)
                pc += 1

            else:
                pc += 1

        t1 = time.perf_counter()
        return ExecutionResult(
            return_value=ret_val,
            variables=env,
            dynamic_instruction_count=step_count,
            simulated_cycles=total_cycles,
            execution_time_sec=t1 - t0,
            terminated_normally=(step_count < self.max_steps),
        )
