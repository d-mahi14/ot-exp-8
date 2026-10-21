"""Intermediate Representation (IR) and Control Flow Graph (CFG) for Track F Mini-Compiler.

Defines:
1. Three-Address Code (TAC) Instructions:
   - Assignments: x = y, x = c
   - Binary ops: x = y op z (+, -, *, /, %)
   - Comparison ops: x = y cmp z (<, <=, >, >=, ==, !=)
   - Unconditional jump: jmp label
   - Conditional jump: jmp_if cond label, jmp_if_false cond label
   - Labels: label:
   - Returns: ret val
2. BasicBlock representation: linear sequence of instructions with single entry and exit.
3. ControlFlowGraph (CFG):
   - Predecessors, successors
   - Dominator tree computation (for natural loop detection and LICM)
   - Natural loop identification
4. Liveness Analysis:
   - Gen/Kill sets per instruction and basic block
   - In/Out live variable sets per block
   - Variable live ranges for register allocation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union


class OpCode(Enum):
    CONST = "const"        # dest = val
    COPY = "copy"          # dest = src
    ADD = "add"            # dest = src1 + src2
    SUB = "sub"            # dest = src1 - src2
    MUL = "mul"            # dest = src1 * src2
    DIV = "div"            # dest = src1 / src2
    MOD = "mod"            # dest = src1 % src2
    LT = "lt"              # dest = src1 < src2
    LE = "le"              # dest = src1 <= src2
    GT = "gt"              # dest = src1 > src2
    GE = "ge"              # dest = src1 >= src2
    EQ = "eq"              # dest = src1 == src2
    NE = "ne"              # dest = src1 != src2
    JMP = "jmp"            # jmp label
    JMP_IF = "jmp_if"      # jmp_if cond label
    JMP_IF_FALSE = "jmp_if_false"  # jmp_if_false cond label
    LABEL = "label"        # label:
    RET = "ret"            # ret val
    PHI = "phi"            # phi node (for SSA)
    SPILL_STORE = "spill_store"  # spill slot = reg
    SPILL_LOAD = "spill_load"    # reg = spill slot


@dataclass
class Instruction:
    """Three-Address Code instruction."""
    op: OpCode
    dest: Optional[str] = None
    src1: Optional[Union[str, int, float]] = None
    src2: Optional[Union[str, int, float]] = None
    label: Optional[str] = None

    def uses(self) -> Set[str]:
        """Return the set of variables read by this instruction."""
        used = set()
        if isinstance(self.src1, str) and not self.src1.isdigit() and not (self.src1.startswith("-") and self.src1[1:].isdigit()):
            used.add(self.src1)
        if isinstance(self.src2, str) and not self.src2.isdigit() and not (self.src2.startswith("-") and self.src2[1:].isdigit()):
            used.add(self.src2)
        return used

    def defs(self) -> Set[str]:
        """Return the set of variables defined (written) by this instruction."""
        if self.dest is not None:
            return {self.dest}
        return set()

    def __repr__(self) -> str:
        if self.op == OpCode.LABEL:
            return f"{self.label}:"
        elif self.op == OpCode.CONST:
            return f"  {self.dest} = {self.src1}"
        elif self.op == OpCode.COPY:
            return f"  {self.dest} = {self.src1}"
        elif self.op in {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MOD,
                         OpCode.LT, OpCode.LE, OpCode.GT, OpCode.GE, OpCode.EQ, OpCode.NE}:
            sym = {
                OpCode.ADD: "+", OpCode.SUB: "-", OpCode.MUL: "*", OpCode.DIV: "/", OpCode.MOD: "%",
                OpCode.LT: "<", OpCode.LE: "<=", OpCode.GT: ">", OpCode.GE: ">=", OpCode.EQ: "==", OpCode.NE: "!="
            }[self.op]
            return f"  {self.dest} = {self.src1} {sym} {self.src2}"
        elif self.op == OpCode.JMP:
            return f"  jmp {self.label}"
        elif self.op == OpCode.JMP_IF:
            return f"  jmp_if {self.src1} {self.label}"
        elif self.op == OpCode.JMP_IF_FALSE:
            return f"  jmp_if_false {self.src1} {self.label}"
        elif self.op == OpCode.RET:
            return f"  ret {self.src1}"
        elif self.op == OpCode.SPILL_STORE:
            return f"  [spill_{self.dest}] = {self.src1}"
        elif self.op == OpCode.SPILL_LOAD:
            return f"  {self.dest} = [spill_{self.src1}]"
        return f"  {self.op.name} {self.dest} {self.src1} {self.src2}"


@dataclass
class BasicBlock:
    """A maximal sequence of instructions with single entry and exit points."""
    block_id: int
    name: str
    instructions: List[Instruction] = field(default_factory=list)
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)

    # Liveness sets
    live_use: Set[str] = field(default_factory=set)  # variables used before any def in this block
    live_def: Set[str] = field(default_factory=set)  # variables defined in this block
    live_in: Set[str] = field(default_factory=set)   # live at entry of block
    live_out: Set[str] = field(default_factory=set)  # live at exit of block

    def compute_local_liveness(self) -> None:
        """Compute live_use (Gen) and live_def (Kill) sets for this block."""
        self.live_use = set()
        self.live_def = set()
        for instr in self.instructions:
            for u in instr.uses():
                if u not in self.live_def:
                    self.live_use.add(u)
            for d in instr.defs():
                self.live_def.add(d)


@dataclass
class NaturalLoop:
    """Represents a natural loop in the CFG."""
    header_id: int
    back_edge_source_id: int
    body_blocks: Set[int] = field(default_factory=set)


class ControlFlowGraph:
    """Control Flow Graph representation with dominator tree and loop analysis."""

    def __init__(self, blocks: List[BasicBlock]) -> None:
        self.blocks: Dict[int, BasicBlock] = {b.block_id: b for b in blocks}
        self.entry_id: int = blocks[0].block_id if blocks else 0
        self.dominators: Dict[int, Set[int]] = {}
        self.natural_loops: List[NaturalLoop] = []

    def compute_liveness(self) -> None:
        r"""Compute backward dataflow iterative liveness analysis:

        In[B] = Use[B] union (Out[B] \setminus Def[B])
        Out[B] = union_{S in Succ[B]} In[S]
        """
        for b in self.blocks.values():
            b.compute_local_liveness()
            b.live_in = set(b.live_use)
            b.live_out = set()

        changed = True
        while changed:
            changed = False
            # Iterate in reverse topological / post-order
            for b_id in sorted(self.blocks.keys(), reverse=True):
                block = self.blocks[b_id]
                new_out: Set[str] = set()
                for succ_id in block.successors:
                    if succ_id in self.blocks:
                        new_out.update(self.blocks[succ_id].live_in)

                new_in = block.live_use | (new_out - block.live_def)

                if new_in != block.live_in or new_out != block.live_out:
                    block.live_in = new_in
                    block.live_out = new_out
                    changed = True

    def compute_dominators(self) -> None:
        """Compute dominator sets using fixed-point iteration:

        Dom(Entry) = {Entry}
        Dom(B) = {B} union ( intersection_{P in Pred(B)} Dom(P) )
        """
        all_blocks = set(self.blocks.keys())
        self.dominators = {b_id: set(all_blocks) for b_id in all_blocks}
        self.dominators[self.entry_id] = {self.entry_id}

        changed = True
        while changed:
            changed = False
            for b_id in self.blocks:
                if b_id == self.entry_id:
                    continue
                preds = [p for p in self.blocks[b_id].predecessors if p in self.blocks]
                if not preds:
                    new_dom = {b_id}
                else:
                    intersection = set(self.dominators[preds[0]])
                    for p in preds[1:]:
                        intersection &= self.dominators[p]
                    new_dom = {b_id} | intersection

                if new_dom != self.dominators[b_id]:
                    self.dominators[b_id] = new_dom
                    changed = True

    def find_natural_loops(self) -> List[NaturalLoop]:
        """Identify natural loops via back-edges (u -> v where v dominates u)."""
        self.compute_dominators()
        loops = []

        for u_id, block in self.blocks.items():
            for v_id in block.successors:
                # If v dominates u, then u -> v is a back edge to loop header v
                if v_id in self.dominators.get(u_id, set()):
                    # Reconstruct loop body: all blocks that can reach u without going through v
                    loop_body = {v_id, u_id}
                    stack = [u_id]
                    while stack:
                        curr = stack.pop()
                        for p_id in self.blocks[curr].predecessors:
                            if p_id not in loop_body:
                                loop_body.add(p_id)
                                stack.append(p_id)
                    loops.append(NaturalLoop(header_id=v_id, back_edge_source_id=u_id, body_blocks=loop_body))

        self.natural_loops = loops
        return loops

    def to_instructions(self) -> List[Instruction]:
        """Flatten basic blocks back into a linear instruction stream."""
        instrs: List[Instruction] = []
        for b_id in sorted(self.blocks.keys()):
            instrs.extend(self.blocks[b_id].instructions)
        return instrs


def build_cfg(instructions: List[Instruction]) -> ControlFlowGraph:
    """Partition a linear instruction stream into basic blocks and link control edges."""
    if not instructions:
        return ControlFlowGraph([])

    # 1. Identify block leader indices
    leaders = {0}
    label_to_index: Dict[str, int] = {}

    for idx, instr in enumerate(instructions):
        if instr.op == OpCode.LABEL:
            leaders.add(idx)
            if instr.label:
                label_to_index[instr.label] = idx
        elif instr.op in {OpCode.JMP, OpCode.JMP_IF, OpCode.JMP_IF_FALSE, OpCode.RET}:
            if idx + 1 < len(instructions):
                leaders.add(idx + 1)

    sorted_leaders = sorted(leaders)
    blocks: List[BasicBlock] = []

    for i, start_idx in enumerate(sorted_leaders):
        end_idx = sorted_leaders[i + 1] if i + 1 < len(sorted_leaders) else len(instructions)
        blk_instrs = instructions[start_idx:end_idx]
        name = f"block_{i}"
        # If block starts with a label, name it after the label
        if blk_instrs and blk_instrs[0].op == OpCode.LABEL and blk_instrs[0].label:
            name = blk_instrs[0].label
        blocks.append(BasicBlock(block_id=i, name=name, instructions=blk_instrs))

    # Map labels to block_ids
    label_to_block_id: Dict[str, int] = {}
    for b in blocks:
        for instr in b.instructions:
            if instr.op == OpCode.LABEL and instr.label:
                label_to_block_id[instr.label] = b.block_id

    # 2. Add predecessor and successor edges
    for i, b in enumerate(blocks):
        if not b.instructions:
            continue
        last_instr = b.instructions[-1]
        if last_instr.op == OpCode.JMP:
            if last_instr.label and last_instr.label in label_to_block_id:
                target_id = label_to_block_id[last_instr.label]
                b.successors.add(target_id)
                blocks[target_id].predecessors.add(b.block_id)
        elif last_instr.op in {OpCode.JMP_IF, OpCode.JMP_IF_FALSE}:
            # Target edge
            if last_instr.label and last_instr.label in label_to_block_id:
                target_id = label_to_block_id[last_instr.label]
                b.successors.add(target_id)
                blocks[target_id].predecessors.add(b.block_id)
            # Fall-through edge
            if i + 1 < len(blocks):
                b.successors.add(blocks[i + 1].block_id)
                blocks[i + 1].predecessors.add(b.block_id)
        elif last_instr.op == OpCode.RET:
            pass  # No successors for return
        else:
            # Fall-through to next sequential block
            if i + 1 < len(blocks):
                b.successors.add(blocks[i + 1].block_id)
                blocks[i + 1].predecessors.add(b.block_id)

    cfg = ControlFlowGraph(blocks)
    cfg.compute_liveness()
    return cfg
