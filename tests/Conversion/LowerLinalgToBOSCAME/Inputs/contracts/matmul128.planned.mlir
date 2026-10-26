// Hand-authored expected plan; not output of an implemented planner.
// Complete snapshot/terminal-store forwarding: 5 source tasks -> 1 physical task.
// 32 programs, each two ordered signed-i8 MMA operations; 0 SPM and 0 DMA.
module attributes {
  contract.version = 1 : i32,
  contract.producer = "neutral-design-reference",
  contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true},
  plan.version = 1 : i32,
  plan.target_profile = "nr-fpga"
} {
  func.func @matmul128(
      %a: memref<*xi8> {contract.value_id = 0 : i64},
      %b: memref<*xi8> {contract.value_id = 1 : i64},
      %c: memref<*xi32> {contract.value_id = 2 : i64},
      %pid_m: index {contract.program = {role = "program_id", axis = 0 : i32}},
      %pid_n: index {contract.program = {role = "program_id", axis = 1 : i32}})
      attributes {
        contract.grid = array<i32: 2, 16, 1>,
        contract.compute_graph = {version = 1 : i32, root_scope = 0 : i64,
          complete = true, scope_count = 1 : i64, source_lineage_complete = true},
        plan.budget = {spm_banks = 2 : i64, bank_capacity = 131072 : i64,
          high_water = array<i64: 0, 0>},
        plan.source_task_count = 5 : i64,
        plan.task_map = [
          {old_id = 0 : i64, new_id = 0 : i64, action = "folded-load"},
          {old_id = 1 : i64, new_id = 0 : i64, action = "folded-load"},
          {old_id = 2 : i64, new_id = 0 : i64, action = "folded-load"},
          {old_id = 3 : i64, new_id = 0 : i64, action = "retained"},
          {old_id = 4 : i64, new_id = 0 : i64, action = "folded-store"}]
      } {
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %row = arith.muli %pid_m, %c64 : index
    %col = arith.muli %pid_n, %c64 : index
    %a_off = arith.muli %row, %c128 : index
    %b_off = arith.muli %col, %c128 : index
    %c_row = arith.muli %row, %c1024 : index
    %c_off = arith.addi %c_row, %col : index
    %av = memref.reinterpret_cast %a to offset: [%a_off], sizes: [64, 128], strides: [128, 1]
      : memref<*xi8> to memref<64x128xi8, strided<[128, 1], offset: ?>>
    %bv = memref.reinterpret_cast %b to offset: [%b_off], sizes: [128, 64], strides: [1, 128]
      : memref<*xi8> to memref<128x64xi8, strided<[1, 128], offset: ?>>
    %cv = memref.reinterpret_cast %c to offset: [%c_off], sizes: [64, 64], strides: [1024, 1]
      : memref<*xi32> to memref<64x64xi32, strided<[1024, 1], offset: ?>>
    linalg.matmul {cast = #linalg.type_fn<cast_signed>,
      contract.task = {id = 0 : i64, kind = "dot", scope = 0 : i64, stage = 0 : i64,
        predecessors = array<i64>, input_values = array<i64: 0, 1, 2>,
        output_values = array<i64>, barrier_after = true, rematerializable = false,
        origin_id = 3 : i64},
      contract.compute = {kind = "dot", signed = true, lhs = "i8", rhs = "i8", acc = "i32"},
      contract.provenance = {origin_id = 3 : i64, source_load_origin_ids = array<i64: 0, 1, 2>,
        source_store_origin_ids = array<i64: 4>,
        a_origin_id = 0 : i64, b_origin_id = 1 : i64, accumulator_origin_id = 2 : i64,
        a_source_load_origin_ids = array<i64: 0>, b_source_load_origin_ids = array<i64: 1>,
        accumulator_source_load_origin_ids = array<i64: 2>,
        a_source_lineage_complete = true, b_source_lineage_complete = true,
        accumulator_source_lineage_complete = true},
      plan.route = {engine = "ame", tile = {m = 64 : i64, n = 64 : i64, k = 64 : i64},
        subtiles = {n = 1 : i64, k = 2 : i64},
        a = {operand = 0 : i64, level = "ddr"},
        b = {operand = 1 : i64, level = "ddr"},
        accumulator = {operand = 2 : i64, level = "ddr"},
        output = {operand = 2 : i64, level = "ddr"}}}
      ins(%av, %bv : memref<64x128xi8, strided<[128, 1], offset: ?>>,
                     memref<128x64xi8, strided<[1, 128], offset: ?>>)
      outs(%cv : memref<64x64xi32, strided<[1024, 1], offset: ?>>)
    return
  }
}
