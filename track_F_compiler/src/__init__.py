"""Track F: Compiler Optimization Module."""

from track_F_compiler.src.compiler import CompilationResult, IRBuilder, MiniCompiler
from track_F_compiler.src.interpreter import ExecutionResult, IRInterpreter
from track_F_compiler.src.ir import (
    BasicBlock,
    ControlFlowGraph,
    Instruction,
    NaturalLoop,
    OpCode,
    build_cfg,
)
from track_F_compiler.src.optimizations import (
    RegisterAllocationResult,
    run_common_subexpression_elimination,
    run_constant_folding_and_propagation,
    run_dead_code_elimination,
    run_loop_invariant_code_motion,
    solve_ilp_register_allocation,
)

__all__ = [
    "CompilationResult",
    "IRBuilder",
    "MiniCompiler",
    "ExecutionResult",
    "IRInterpreter",
    "BasicBlock",
    "ControlFlowGraph",
    "Instruction",
    "NaturalLoop",
    "OpCode",
    "build_cfg",
    "RegisterAllocationResult",
    "run_common_subexpression_elimination",
    "run_constant_folding_and_propagation",
    "run_dead_code_elimination",
    "run_loop_invariant_code_motion",
    "solve_ilp_register_allocation",
]
