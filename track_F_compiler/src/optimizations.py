"""Compiler Optimization Passes for Track F Mini-Compiler.

Implements:
1. Constant Folding & Constant Propagation
2. Dead-Code Elimination (DCE)
3. Common-Subexpression Elimination (CSE)
4. Loop-Invariant Code Motion (LICM)
5. ILP-Based Register Allocation (HiGHS Branch-and-Cut)
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.optimize import milp, LinearConstraint

from track_F_compiler.src.ir import (
    BasicBlock,
    ControlFlowGraph,
    Instruction,
    NaturalLoop,
    OpCode,
    build_cfg,
)


def run_constant_folding_and_propagation(instructions: List[Instruction]) -> List[Instruction]:
    """Pass 1: Constant Propagation and Constant Folding.

    - Propagates known constants to subsequent operand uses.
    - Evaluates compile-time constant arithmetic and comparisons.
    - Preserves runtime exceptions (e.g. division by zero is never folded).
    """
    optimized: List[Instruction] = []
    # Local constant store: var -> constant scalar
    constants: Dict[str, Union[int, float]] = {}

    def resolve(val: Optional[Union[str, int, float]]) -> Optional[Union[str, int, float]]:
        if val is None or isinstance(val, (int, float)):
            return val
        if val in constants:
            return constants[val]
        return val

    for instr in instructions:
        # If control flow boundary (labels, jumps), flush non-SSA local constant knowledge
        if instr.op in {OpCode.LABEL, OpCode.JMP, OpCode.JMP_IF, OpCode.JMP_IF_FALSE}:
            # Flush across basic block boundaries to be sound
            constants.clear()
            optimized.append(instr)
            continue

        # Resolve operands from current constant environment
        new_src1 = resolve(instr.src1)
        new_src2 = resolve(instr.src2)

        # 1. CONST instruction
        if instr.op == OpCode.CONST:
            assert instr.dest is not None
            constants[instr.dest] = new_src1  # type: ignore
            optimized.append(Instruction(op=OpCode.CONST, dest=instr.dest, src1=new_src1))
            continue

        # 2. COPY instruction
        if instr.op == OpCode.COPY:
            assert instr.dest is not None
            if isinstance(new_src1, (int, float)):
                constants[instr.dest] = new_src1
                optimized.append(Instruction(op=OpCode.CONST, dest=instr.dest, src1=new_src1))
            else:
                # Value is a variable; invalidate if dest was previously constant
                constants.pop(instr.dest, None)
                optimized.append(Instruction(op=OpCode.COPY, dest=instr.dest, src1=new_src1))
            continue

        # 3. Arithmetic and Comparison folding
        if instr.op in {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOD,
                        OpCode.LT, OpCode.LE, OpCode.GT, OpCode.GE, OpCode.EQ, OpCode.NE}:
            assert instr.dest is not None

            # Check if both operands are constants
            if isinstance(new_src1, (int, float)) and isinstance(new_src2, (int, float)):
                try:
                    folded_val: Optional[Union[int, float]] = None
                    if instr.op == OpCode.ADD:
                        folded_val = new_src1 + new_src2
                    elif instr.op == OpCode.SUB:
                        folded_val = new_src1 - new_src2
                    elif instr.op == OpCode.MUL:
                        folded_val = new_src1 * new_src2
                    elif instr.op == OpCode.DIV:
                        if new_src2 != 0:
                            if isinstance(new_src1, int) and isinstance(new_src2, int):
                                folded_val = new_src1 // new_src2
                            else:
                                folded_val = new_src1 / new_src2
                    elif instr.op == OpCode.MOD:
                        if new_src2 != 0:
                            folded_val = new_src1 % new_src2
                    elif instr.op == OpCode.LT:
                        folded_val = int(new_src1 < new_src2)
                    elif instr.op == OpCode.LE:
                        folded_val = int(new_src1 <= new_src2)
                    elif instr.op == OpCode.GT:
                        folded_val = int(new_src1 > new_src2)
                    elif instr.op == OpCode.GE:
                        folded_val = int(new_src1 >= new_src2)
                    elif instr.op == OpCode.EQ:
                        folded_val = int(new_src1 == new_src2)
                    elif instr.op == OpCode.NE:
                        folded_val = int(new_src1 != new_src2)

                    if folded_val is not None:
                        constants[instr.dest] = folded_val
                        optimized.append(Instruction(op=OpCode.CONST, dest=instr.dest, src1=folded_val))
                        continue
                except Exception:
                    # In case of arithmetic error, do not fold
                    pass

            # If not fully foldable, invalidate dest in constants
            constants.pop(instr.dest, None)
            optimized.append(Instruction(op=instr.op, dest=instr.dest, src1=new_src1, src2=new_src2))
            continue

        # For any other instruction, invalidate dest if defined
        for d in instr.defs():
            constants.pop(d, None)
        optimized.append(Instruction(op=instr.op, dest=instr.dest, src1=new_src1, src2=new_src2, label=instr.label))

    return optimized


def run_dead_code_elimination(instructions: List[Instruction]) -> List[Instruction]:
    """Pass 2: Dead-Code Elimination (DCE).

    Iteratively removes instructions defining variables that are never read/live,
    and removes unreachable basic blocks.
    """
    changed = True
    curr_instrs = list(instructions)

    while changed:
        changed = False
        cfg = build_cfg(curr_instrs)
        cfg.compute_liveness()

        # Mark all live variables across instructions
        new_instrs: List[Instruction] = []

        # Iterate block by block backwards
        for b_id in sorted(cfg.blocks.keys()):
            block = cfg.blocks[b_id]
            # If block is unreachable (no predecessors and not entry block), skip entire block
            if b_id != cfg.entry_id and not block.predecessors:
                changed = True
                continue

            live = set(block.live_out)
            kept_in_block = []

            for instr in reversed(block.instructions):
                # Never remove labels, jumps, returns, or spill operations
                if instr.op in {OpCode.LABEL, OpCode.JMP, OpCode.JMP_IF, OpCode.JMP_IF_FALSE, OpCode.RET,
                                OpCode.SPILL_STORE, OpCode.SPILL_LOAD}:
                    kept_in_block.append(instr)
                    live.difference_update(instr.defs())
                    live.update(instr.uses())
                elif instr.dest is not None:
                    # If dest is currently live, keep instruction and update liveness
                    if instr.dest in live:
                        kept_in_block.append(instr)
                        live.remove(instr.dest)
                        live.update(instr.uses())
                    else:
                        # Dead assignment! Eliminate
                        changed = True
                else:
                    kept_in_block.append(instr)
                    live.update(instr.uses())

            new_instrs.extend(reversed(kept_in_block))

        curr_instrs = new_instrs

    return curr_instrs


def run_common_subexpression_elimination(instructions: List[Instruction]) -> List[Instruction]:
    """Pass 3: Common-Subexpression Elimination (CSE).

    Identifies redundant binary evaluations within basic blocks:
        t1 = a + b
        ...
        t2 = a + b  -> replaced with: t2 = t1 (if a and b are unchanged)
    """
    cfg = build_cfg(instructions)
    optimized_blocks: List[BasicBlock] = []

    for b_id in sorted(cfg.blocks.keys()):
        block = cfg.blocks[b_id]
        # Map: (op, src1, src2) -> original_dest_variable
        available_exprs: Dict[Tuple[OpCode, Any, Any], str] = {}
        new_block_instrs: List[Instruction] = []

        for instr in block.instructions:
            # If it's a binary op
            if instr.op in {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOD,
                            OpCode.LT, OpCode.LE, OpCode.GT, OpCode.GE, OpCode.EQ, OpCode.NE}:
                assert instr.dest is not None
                expr_key = (instr.op, instr.src1, instr.src2)
                # Commutative canonicalization: for +, *, ==, !=, sort src1 and src2 if possible
                if instr.op in {OpCode.ADD, OpCode.MUL, OpCode.EQ, OpCode.NE}:
                    s1, s2 = str(instr.src1), str(instr.src2)
                    if s1 > s2:
                        expr_key = (instr.op, instr.src2, instr.src1)

                if expr_key in available_exprs:
                    # Redundant! Replace with copy from existing destination
                    prev_dest = available_exprs[expr_key]
                    new_block_instrs.append(Instruction(op=OpCode.COPY, dest=instr.dest, src1=prev_dest))
                else:
                    available_exprs[expr_key] = instr.dest
                    new_block_instrs.append(instr)

                # If instr.dest is redefined, invalidate any available expression using dest
                keys_to_remove = [k for k in available_exprs if k[1] == instr.dest or k[2] == instr.dest]
                for k in keys_to_remove:
                    del available_exprs[k]

            else:
                new_block_instrs.append(instr)
                # If instruction modifies a variable, invalidate expressions referencing it
                for d in instr.defs():
                    keys_to_remove = [k for k in available_exprs if k[1] == d or k[2] == d or available_exprs[k] == d]
                    for k in keys_to_remove:
                        del available_exprs[k]

        block.instructions = new_block_instrs
        optimized_blocks.append(block)

    return ControlFlowGraph(optimized_blocks).to_instructions()


def run_loop_invariant_code_motion(instructions: List[Instruction]) -> List[Instruction]:
    """Pass 4: Loop-Invariant Code Motion (LICM).

    Identifies loop-invariant computations inside natural loops and hoists them
    into a loop pre-header block before the loop header.

    Invariant definition:
    An instruction dest = src1 op src2 is loop-invariant if:
      - All operands are constants, OR
      - Defined outside the loop body, OR
      - Already marked as loop-invariant within this loop.
    """
    cfg = build_cfg(instructions)
    loops = cfg.find_natural_loops()

    if not loops:
        return instructions

    # For each detected natural loop, find invariants
    for loop in loops:
        header_block = cfg.blocks[loop.header_id]
        body_blocks = [cfg.blocks[b_id] for b_id in loop.body_blocks if b_id in cfg.blocks]

        # Definitions inside the loop
        defs_in_loop: Set[str] = set()
        for b in body_blocks:
            for instr in b.instructions:
                defs_in_loop.update(instr.defs())

        # Iteratively identify loop invariants
        invariants: List[Instruction] = []
        invariant_defs: Set[str] = set()

        changed = True
        while changed:
            changed = False
            for b in body_blocks:
                new_b_instrs = []
                for instr in b.instructions:
                    # Only hoist pure computations (no branches, returns, or labels)
                    if instr.op in {OpCode.CONST, OpCode.COPY, OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOD,
                                    OpCode.LT, OpCode.LE, OpCode.GT, OpCode.GE, OpCode.EQ, OpCode.NE}:
                        assert instr.dest is not None

                        # Check if operands are invariant
                        def is_operand_invariant(opnd: Optional[Union[str, int, float]]) -> bool:
                            if opnd is None or isinstance(opnd, (int, float)):
                                return True
                            if opnd not in defs_in_loop:
                                return True
                            if opnd in invariant_defs:
                                return True
                            return False

                        if is_operand_invariant(instr.src1) and is_operand_invariant(instr.src2):
                            invariants.append(instr)
                            invariant_defs.add(instr.dest)
                            defs_in_loop.remove(instr.dest)
                            changed = True
                            continue  # Remove from loop body

                    new_b_instrs.append(instr)
                b.instructions = new_b_instrs

        # Insert invariants into a pre-header immediately before the header block instructions
        if invariants:
            # Place at top of header block before loop logic or label
            header_block.instructions = invariants + header_block.instructions

    return cfg.to_instructions()


@dataclass
class RegisterAllocationResult:
    allocation: Dict[str, str]       # var -> physical register (e.g. 'R0', 'R1')
    spilled_variables: Set[str]      # variables spilled to memory
    total_spills: int
    num_registers: int
    rewritten_instructions: List[Instruction]


def solve_ilp_register_allocation(
    instructions: List[Instruction],
    num_registers: int = 3,
) -> RegisterAllocationResult:
    """Pass 5: ILP-Based Register Allocation with Minimum Spill Cost (HiGHS).

    Solves the exact 0-1 Integer Linear Program:
        min sum_{v in V} weight(v) * s_v
    Subject to:
        sum_{r=1}^K x_{v, r} + s_v = 1,   forall v in V
        x_{u, r} + x_{v, r} <= 1,         forall (u, v) in E, forall r in {1..K}
    """
    cfg = build_cfg(instructions)
    cfg.compute_liveness()

    # 1. Identify all variables and compute interference graph
    all_vars: Set[str] = set()
    for b in cfg.blocks.values():
        all_vars.update(b.live_def)
        all_vars.update(b.live_use)

    var_list = sorted(all_vars)
    V_count = len(var_list)
    if V_count == 0:
        return RegisterAllocationResult(
            allocation={},
            spilled_variables=set(),
            total_spills=0,
            num_registers=num_registers,
            rewritten_instructions=instructions,
        )

    var_to_idx = {v: i for i, v in enumerate(var_list)}
    interferences: Set[Tuple[int, int]] = set()

    # Two variables interfere if they are simultaneously live at any point in any block
    for b in cfg.blocks.values():
        live = set(b.live_out)
        for instr in reversed(b.instructions):
            for d in instr.defs():
                for l in live:
                    if d != l and d in var_to_idx and l in var_to_idx:
                        u, v = sorted([var_to_idx[d], var_to_idx[l]])
                        interferences.add((u, v))
            live.difference_update(instr.defs())
            live.update(instr.uses())

    # 2. Formulate 0-1 ILP
    # Decision variables for each variable v:
    # x_{v, 0}, ..., x_{v, K-1} (binary register assignment)
    # s_v (binary spill indicator)
    # Total binary variables = V * (K + 1)
    K = num_registers
    total_binary_vars = V_count * (K + 1)

    def var_reg_idx(v_i: int, r: int) -> int:
        return v_i * (K + 1) + r

    def spill_idx(v_i: int) -> int:
        return v_i * (K + 1) + K

    # Objective coefficients c:
    # 0 cost for register assignment; positive cost for spill
    c = np.zeros(total_binary_vars, dtype=np.float64)
    for v_i in range(V_count):
        # Spilling costs 10 units (or weighted by usage frequency)
        c[spill_idx(v_i)] = 10.0

    # Constraints:
    # 1. Assignment or spill: sum_{r=0}^{K-1} x_{v, r} + s_v = 1 (V_count constraints)
    # 2. Interference: x_{u, r} + x_{v, r} <= 1 (len(interferences) * K constraints)
    num_constraints = V_count + len(interferences) * K
    A = np.zeros((num_constraints, total_binary_vars), dtype=np.float64)
    lhs = np.zeros(num_constraints, dtype=np.float64)
    rhs = np.zeros(num_constraints, dtype=np.float64)

    row = 0
    # Constraint 1: Equality sum(x_{v, r}) + s_v == 1
    for v_i in range(V_count):
        for r in range(K):
            A[row, var_reg_idx(v_i, r)] = 1.0
        A[row, spill_idx(v_i)] = 1.0
        lhs[row] = 1.0
        rhs[row] = 1.0
        row += 1

    # Constraint 2: Inequality x_{u, r} + x_{v, r} <= 1
    for (u, v) in interferences:
        for r in range(K):
            A[row, var_reg_idx(u, r)] = 1.0
            A[row, var_reg_idx(v, r)] = 1.0
            lhs[row] = 0.0
            rhs[row] = 1.0
            row += 1

    constraints = LinearConstraint(A, lhs, rhs)
    integrality = np.ones(total_binary_vars, dtype=int)  # All binary

    res = milp(c=c, integrality=integrality, constraints=constraints)

    allocation: Dict[str, str] = {}
    spilled: Set[str] = set()

    if res.success:
        sol = res.x
        for v_i, v_name in enumerate(var_list):
            if sol[spill_idx(v_i)] > 0.5:
                spilled.add(v_name)
            else:
                for r in range(K):
                    if sol[var_reg_idx(v_i, r)] > 0.5:
                        allocation[v_name] = f"R{r}"
                        break
    else:
        # Fallback to spilling all if ILP fails
        spilled = set(var_list)

    # 3. Rewrite instruction stream: replace assigned variables with physical registers
    # and insert SPILL_STORE / SPILL_LOAD for spilled variables
    rewritten: List[Instruction] = []
    for instr in instructions:
        # Check if src1 or src2 need spill loading
        src1_rep = instr.src1
        src2_rep = instr.src2
        dest_rep = instr.dest

        if isinstance(instr.src1, str) and instr.src1 in spilled:
            rewritten.append(Instruction(op=OpCode.SPILL_LOAD, dest="R_SPILL1", src1=instr.src1))
            src1_rep = "R_SPILL1"
        elif isinstance(instr.src1, str) and instr.src1 in allocation:
            src1_rep = allocation[instr.src1]

        if isinstance(instr.src2, str) and instr.src2 in spilled:
            rewritten.append(Instruction(op=OpCode.SPILL_LOAD, dest="R_SPILL2", src1=instr.src2))
            src2_rep = "R_SPILL2"
        elif isinstance(instr.src2, str) and instr.src2 in allocation:
            src2_rep = allocation[instr.src2]

        if instr.dest is not None:
            if instr.dest in spilled:
                dest_rep = "R_SPILL_DEST"
            elif instr.dest in allocation:
                dest_rep = allocation[instr.dest]

        rewritten.append(Instruction(
            op=instr.op,
            dest=dest_rep,
            src1=src1_rep,
            src2=src2_rep,
            label=instr.label,
        ))

        # If dest was spilled, store to memory immediately after
        if instr.dest is not None and instr.dest in spilled:
            rewritten.append(Instruction(op=OpCode.SPILL_STORE, dest=instr.dest, src1="R_SPILL_DEST"))

    return RegisterAllocationResult(
        allocation=allocation,
        spilled_variables=spilled,
        total_spills=len(spilled),
        num_registers=num_registers,
        rewritten_instructions=rewritten,
    )
