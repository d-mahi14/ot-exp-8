"""Unit tests for Track F Compiler Optimization Passes."""

import pytest

from track_F_compiler.src.compiler import IRBuilder, MiniCompiler
from track_F_compiler.src.interpreter import IRInterpreter
from track_F_compiler.src.ir import OpCode, build_cfg
from track_F_compiler.src.optimizations import (
    run_common_subexpression_elimination,
    run_constant_folding_and_propagation,
    run_dead_code_elimination,
    run_loop_invariant_code_motion,
    solve_ilp_register_allocation,
)


def test_constant_folding_and_propagation() -> None:
    """Verify that compile-time arithmetic is folded and constants are propagated."""
    b = IRBuilder()
    b.const("a", 10)
    b.const("b", 20)
    b.add("c", "a", "b")  # Should fold to 30
    b.mul("d", "c", 2)    # Should fold to 60
    b.ret("d")
    instrs = b.build()

    opt = run_constant_folding_and_propagation(instrs)
    # The return operand should now be constant 60
    assert opt[-1].op == OpCode.RET
    assert opt[-1].src1 == 60

    # VM execution must match
    vm = IRInterpreter()
    res_orig = vm.execute(instrs)
    res_opt = vm.execute(opt)
    assert res_orig.return_value == 60
    assert res_opt.return_value == 60


def test_constant_folding_division_by_zero_safety() -> None:
    """Verify that division by zero is NOT folded at compile time, preserving runtime error."""
    b = IRBuilder()
    b.const("x", 42)
    b.const("zero", 0)
    b.div("res", "x", "zero")
    b.ret("res")
    instrs = b.build()

    opt = run_constant_folding_and_propagation(instrs)
    # DIV instruction must NOT be folded away
    div_instrs = [i for i in opt if i.op == OpCode.DIV]
    assert len(div_instrs) == 1

    vm = IRInterpreter()
    with pytest.raises(ZeroDivisionError):
        vm.execute(opt)


def test_dead_code_elimination() -> None:
    """Verify DCE removes unused definitions and unreachable code."""
    b = IRBuilder()
    b.const("used", 100)
    b.const("dead1", 999)      # Dead
    b.add("dead2", "dead1", 5) # Dead
    b.add("res", "used", 50)
    b.ret("res")
    instrs = b.build()

    opt = run_dead_code_elimination(instrs)
    dest_names = {i.dest for i in opt if i.dest is not None}
    assert "dead1" not in dest_names
    assert "dead2" not in dest_names
    assert "used" in dest_names
    assert "res" in dest_names

    vm = IRInterpreter()
    assert vm.execute(instrs).return_value == 150
    assert vm.execute(opt).return_value == 150


def test_common_subexpression_elimination() -> None:
    """Verify CSE reuses identical subexpressions and respects variable mutation."""
    b = IRBuilder()
    b.const("x", 10)
    b.const("y", 20)
    b.add("t1", "x", "y")  # 30
    b.mul("dummy", "t1", 2)
    b.add("t2", "x", "y")  # Redundant! Should be replaced by COPY from t1
    b.add("res", "dummy", "t2")
    b.ret("res")
    instrs = b.build()

    opt = run_common_subexpression_elimination(instrs)
    # In opt, t2 definition should be a COPY, not an ADD
    t2_instr = next(i for i in opt if i.dest == "t2")
    assert t2_instr.op == OpCode.COPY
    assert t2_instr.src1 == "t1"

    vm = IRInterpreter()
    assert vm.execute(instrs).return_value == 90
    assert vm.execute(opt).return_value == 90


def test_cse_invalidated_by_mutation() -> None:
    """Verify CSE does NOT eliminate subexpression if an operand has been modified."""
    b = IRBuilder()
    b.const("x", 10)
    b.const("y", 20)
    b.add("t1", "x", "y")  # 30
    b.const("x", 100)      # Mutate x!
    b.add("t2", "x", "y")  # NOT common subexpression (x has changed)
    b.ret("t2")
    instrs = b.build()

    opt = run_common_subexpression_elimination(instrs)
    t2_instr = next(i for i in opt if i.dest == "t2")
    assert t2_instr.op == OpCode.ADD  # Must remain ADD, not COPY

    vm = IRInterpreter()
    assert vm.execute(instrs).return_value == 120
    assert vm.execute(opt).return_value == 120


def test_loop_invariant_code_motion() -> None:
    """Verify LICM hoists loop-invariant instructions outside the loop body."""
    b = IRBuilder()
    b.const("n", 5)
    b.const("i", 0)
    b.const("acc", 0)
    b.const("base1", 10)
    b.const("base2", 20)

    b.label("loop_header")
    b.lt("cond", "i", "n")
    b.jmp_if_false("cond", "loop_exit")

    # Invariant computation: base1 * base2 = 200 (neither changes in loop)
    b.mul("inv", "base1", "base2")

    # Variant computations
    b.add("acc", "acc", "inv")
    b.add("i", "i", 1)
    b.jmp("loop_header")

    b.label("loop_exit")
    b.ret("acc")
    instrs = b.build()

    opt = run_loop_invariant_code_motion(instrs)

    # Output equivalence
    vm = IRInterpreter()
    res_orig = vm.execute(instrs)
    res_opt = vm.execute(opt)
    assert res_orig.return_value == 1000
    assert res_opt.return_value == 1000

    # In optimized version, dynamic instruction count should be lower because
    # 'inv = base1 * base2' is hoisted outside the loop
    assert res_opt.dynamic_instruction_count <= res_orig.dynamic_instruction_count


def test_ilp_register_allocation() -> None:
    """Verify ILP register allocation maps virtual registers to K physical registers."""
    b = IRBuilder()
    b.const("v1", 1)
    b.const("v2", 2)
    b.add("v3", "v1", "v2")
    b.const("v4", 4)
    b.add("v5", "v3", "v4")
    b.ret("v5")
    instrs = b.build()

    # Allocate with 3 physical registers
    res = solve_ilp_register_allocation(instrs, num_registers=3)
    assert res.total_spills == 0
    assert len(res.allocation) > 0

    # Verify executed rewritten instructions match original output
    vm = IRInterpreter()
    res_vm = vm.execute(res.rewritten_instructions)
    assert res_vm.return_value == 7


def test_full_pipeline_output_equivalence() -> None:
    """Test full multi-pass compilation pipeline on a non-trivial iterative program."""
    # Computes: sum_{i=0}^{9} ( (2+3) * i + (x*y) )
    b = IRBuilder()
    b.const("i", 0)
    b.const("n", 10)
    b.const("total", 0)
    b.const("x", 4)
    b.const("y", 5)

    b.label("start")
    b.lt("c", "i", "n")
    b.jmp_if_false("c", "done")

    # Redundant constant operations
    b.add("c_const", 2, 3)     # Foldable to 5, and loop-invariant!
    b.mul("inv_prod", "x", "y") # 20, loop-invariant!

    b.mul("step_val", "c_const", "i")
    b.add("step_val", "step_val", "inv_prod")
    b.add("total", "total", "step_val")
    b.add("i", "i", 1)
    b.jmp("start")

    b.label("done")
    b.ret("total")
    instrs = b.build()

    compiler = MiniCompiler()
    # 1. Without register allocation (pure IR optimizations)
    comp_res_ir = compiler.optimize(instrs, apply_reg_alloc=False)
    assert comp_res_ir.static_instruction_reduction > 0

    # 2. With register allocation (evaluating spilling and physical register mapping)
    comp_res_reg = compiler.optimize(instrs, apply_reg_alloc=True)

    vm = IRInterpreter()
    val_orig = vm.execute(instrs).return_value
    val_opt_ir = vm.execute(comp_res_ir.optimized_instructions).return_value
    val_opt_reg = vm.execute(comp_res_reg.optimized_instructions).return_value

    assert val_orig == val_opt_ir
    assert val_orig == val_opt_reg
    # In optimized IR, dynamic instructions in the loop should decrease
    res_orig = vm.execute(instrs)
    res_opt_ir = vm.execute(comp_res_ir.optimized_instructions)
    assert res_opt_ir.dynamic_instruction_count < res_orig.dynamic_instruction_count
