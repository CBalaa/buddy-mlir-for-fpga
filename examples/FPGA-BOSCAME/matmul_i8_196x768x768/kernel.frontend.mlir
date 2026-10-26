module attributes {contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}, contract.producer = "triton-riscv", contract.version = 1 : i32} {
  func.func @triton_matmul_i8_196x768x768(%arg0: memref<*xi8> {contract.value_id = 0 : i64}, %arg1: memref<*xi8> {contract.value_id = 1 : i64}, %arg2: memref<*xi32> {contract.value_id = 2 : i64}, %arg3: i32 {contract.program = {axis = 0 : i32, role = "num_programs"}}, %arg4: i32 {contract.program = {axis = 1 : i32, role = "num_programs"}}, %arg5: i32 {contract.program = {axis = 2 : i32, role = "num_programs"}}, %arg6: i32 {contract.program = {axis = 0 : i32, role = "program_id"}}, %arg7: i32 {contract.program = {axis = 1 : i32, role = "program_id"}}, %arg8: i32 {contract.program = {axis = 2 : i32, role = "program_id"}}) attributes {contract.compute_graph = {complete = true, root_scope = 0 : i64, scope_count = 5 : i64, source_lineage_complete = true, version = 1 : i32}, contract.grid = array<i32: 7, 6, 1>, gpu.known_grid_size = array<i32: 7, 6, 1>, llvm.emit_c_interface, noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i8 = arith.constant 0 : i8
    %c768 = arith.constant 768 : index
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c6_i32 = arith.constant 6 : i32
    %0 = arith.cmpi slt, %arg6, %c6_i32 : i32
    scf.if %0 {
      %1 = arith.muli %arg6, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.muli %arg7, %c128_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.muli %2, %c768 : index
      %6 = arith.addi %5, %4 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [32, 128], strides: [768, 1] : memref<*xi32> to memref<32x128xi32, strided<[768, 1], offset: ?>>
      %alloc = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 32, 128>, role = "tile"}} : memref<32x128xi32>
      memref.copy %reinterpret_cast, %alloc {contract.access = {boundary = "external_ddr", kind = "load", mask = {kind = "none"}, origin_id = 0 : i64, other = 0 : i32}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = true, id = 0 : i64, input_values = array<i64: 2>, kind = "memory-load", origin_id = 0 : i64, output_values = array<i64: 3>, predecessors = array<i64>, rematerializable = false, scope = 1 : i64, stage = 0 : i64}} : memref<32x128xi32, strided<[768, 1], offset: ?>> to memref<32x128xi32>
      %7 = arith.muli %4, %c768 : index
      scf.for %arg9 = %c0_i32 to %c6_i32 step %c1_i32  : i32 {
        %8 = arith.muli %arg9, %c128_i32 : i32
        %9 = arith.index_cast %8 : i32 to index
        %10 = arith.addi %5, %9 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 128], strides: [768, 1] : memref<*xi8> to memref<32x128xi8, strided<[768, 1], offset: ?>>
        %alloc_1 = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 32, 128>, role = "tile"}} : memref<32x128xi8>
        memref.copy %reinterpret_cast_0, %alloc_1 {contract.access = {boundary = "external_ddr", kind = "load", mask = {kind = "none"}, origin_id = 1 : i64, other = 0 : i8}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 1>}, contract.task = {barrier_after = true, id = 1 : i64, input_values = array<i64: 0>, kind = "memory-load", origin_id = 1 : i64, output_values = array<i64: 6>, predecessors = array<i64>, rematerializable = false, scope = 2 : i64, stage = 0 : i64}} : memref<32x128xi8, strided<[768, 1], offset: ?>> to memref<32x128xi8>
        %11 = arith.addi %9, %7 : index
        %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [128, 128], strides: [1, 768] : memref<*xi8> to memref<128x128xi8, strided<[1, 768], offset: ?>>
        %alloc_3 = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 128, 128>, role = "tile"}} : memref<16384xi8>
        %reinterpret_cast_4 = memref.reinterpret_cast %alloc_3 to offset: [0], sizes: [128, 128], strides: [1, 128] : memref<16384xi8> to memref<128x128xi8, strided<[1, 128]>>
        memref.copy %reinterpret_cast_2, %reinterpret_cast_4 {contract.access = {boundary = "external_ddr", kind = "load", mask = {kind = "none"}, origin_id = 2 : i64, other = 0 : i8}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 2>}, contract.task = {barrier_after = true, id = 2 : i64, input_values = array<i64: 1>, kind = "memory-load", origin_id = 2 : i64, output_values = array<i64: 7>, predecessors = array<i64>, rematerializable = false, scope = 2 : i64, stage = 0 : i64}} : memref<128x128xi8, strided<[1, 768], offset: ?>> to memref<128x128xi8, strided<[1, 128]>>
        linalg.matmul {contract.provenance = {a_origin_id = 1 : i64, a_source_lineage_complete = true, a_source_load_origin_ids = array<i64: 1>, accumulator_source_lineage_complete = true, accumulator_source_load_origin_ids = array<i64: 0>, b_origin_id = 2 : i64, b_source_lineage_complete = true, b_source_load_origin_ids = array<i64: 2>, origin_id = 3 : i64, source_lineage_complete = true, source_load_origin_ids = array<i64: 0, 1, 2>}, contract.task = {barrier_after = true, id = 3 : i64, input_values = array<i64: 6, 7, 4>, kind = "dot", output_values = array<i64: 8>, predecessors = array<i64: 0, 1, 2>, rematerializable = false, scope = 2 : i64, stage = 1 : i64}, cast = #linalg.type_fn<cast_signed>} ins(%alloc_1, %reinterpret_cast_4 : memref<32x128xi8>, memref<128x128xi8, strided<[1, 128]>>) outs(%alloc : memref<32x128xi32>)
      } {contract.cf = {kind = "uniform_for", serial = true, trip_count = 6 : i64, uniform = true}, contract.loop_carried_slots = [{backedge_producer_compute_ids = array<i64: 3>, exit_consumer_compute_ids = array<i64: 4>, init_producer_compute_ids = array<i64: 0>, init_value_id = 3 : i64, iter_arg_consumer_compute_ids = array<i64: 3>, iter_arg_value_id = 4 : i64, result_value_id = 5 : i64, slot = 0 : i64, yielded_value_id = 8 : i64}], contract.scope = {complete = true, id = 2 : i64, kind = "for-body", parent = 1 : i64}, contract.value_ids = array<i64: 4, 5>}
      memref.copy %alloc, %reinterpret_cast {contract.access = {boundary = "external_ddr", kind = "store", mask = {kind = "none"}, origin_id = 4 : i64}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0, 1, 2>}, contract.task = {barrier_after = true, id = 4 : i64, input_values = array<i64: 5>, kind = "memory-store", origin_id = 4 : i64, output_values = array<i64>, predecessors = array<i64: 0, 3>, rematerializable = false, scope = 1 : i64, stage = 2 : i64}} : memref<32x128xi32> to memref<32x128xi32, strided<[768, 1], offset: ?>>
    } else {
      %1 = arith.muli %arg6, %c32_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.muli %arg7, %c128_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.muli %2, %c768 : index
      %6 = arith.addi %5, %4 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%6], sizes: [32, 128], strides: [768, 1] : memref<*xi32> to memref<32x128xi32, strided<[768, 1], offset: ?>>
      %alloc = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 32, 128>, role = "tile"}} : memref<32x128xi32>
      linalg.fill {contract.fill = {padding = true}, contract.task = {barrier_after = false, id = 5 : i64, input_values = array<i64>, kind = "fill", output_values = array<i64: 9>, predecessors = array<i64>, rematerializable = false, scope = 3 : i64, stage = 0 : i64}} ins(%c0_i32 : i32) outs(%alloc : memref<32x128xi32>)
      %subview = memref.subview %reinterpret_cast[0, 0] [4, 128] [1, 1] : memref<32x128xi32, strided<[768, 1], offset: ?>> to memref<4x128xi32, strided<[768, 1], offset: ?>>
      %subview_0 = memref.subview %alloc[0, 0] [4, 128] [1, 1] : memref<32x128xi32> to memref<4x128xi32, strided<[128, 1]>>
      memref.copy %subview, %subview_0 {contract.access = {boundary = "external_ddr", kind = "load", mask = {kind = "rectangular", valid_shape = array<i64: 4, 128>}, origin_id = 0 : i64, other = 0 : i32}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = true, id = 6 : i64, input_values = array<i64: 2, 9>, kind = "memory-load", origin_id = 0 : i64, output_values = array<i64: 10>, predecessors = array<i64: 5>, rematerializable = false, scope = 3 : i64, stage = 0 : i64}} : memref<4x128xi32, strided<[768, 1], offset: ?>> to memref<4x128xi32, strided<[128, 1]>>
      %7 = arith.muli %4, %c768 : index
      scf.for %arg9 = %c0_i32 to %c6_i32 step %c1_i32  : i32 {
        %8 = arith.muli %arg9, %c128_i32 : i32
        %9 = arith.index_cast %8 : i32 to index
        %10 = arith.addi %5, %9 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 128], strides: [768, 1] : memref<*xi8> to memref<32x128xi8, strided<[768, 1], offset: ?>>
        %alloc_2 = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 32, 128>, role = "tile"}} : memref<32x128xi8>
        linalg.fill {contract.fill = {padding = true}, contract.task = {barrier_after = false, id = 7 : i64, input_values = array<i64>, kind = "fill", output_values = array<i64: 13>, predecessors = array<i64>, rematerializable = false, scope = 4 : i64, stage = 0 : i64}} ins(%c0_i8 : i8) outs(%alloc_2 : memref<32x128xi8>)
        %subview_3 = memref.subview %reinterpret_cast_1[0, 0] [4, 128] [1, 1] : memref<32x128xi8, strided<[768, 1], offset: ?>> to memref<4x128xi8, strided<[768, 1], offset: ?>>
        %subview_4 = memref.subview %alloc_2[0, 0] [4, 128] [1, 1] : memref<32x128xi8> to memref<4x128xi8, strided<[128, 1]>>
        memref.copy %subview_3, %subview_4 {contract.access = {boundary = "external_ddr", kind = "load", mask = {kind = "rectangular", valid_shape = array<i64: 4, 128>}, origin_id = 1 : i64, other = 0 : i8}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 1>}, contract.task = {barrier_after = true, id = 8 : i64, input_values = array<i64: 0, 13>, kind = "memory-load", origin_id = 1 : i64, output_values = array<i64: 14>, predecessors = array<i64: 7>, rematerializable = false, scope = 4 : i64, stage = 0 : i64}} : memref<4x128xi8, strided<[768, 1], offset: ?>> to memref<4x128xi8, strided<[128, 1]>>
        %11 = arith.addi %9, %7 : index
        %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [128, 128], strides: [1, 768] : memref<*xi8> to memref<128x128xi8, strided<[1, 768], offset: ?>>
        %alloc_6 = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 128, 128>, role = "tile"}} : memref<16384xi8>
        %reinterpret_cast_7 = memref.reinterpret_cast %alloc_6 to offset: [0], sizes: [128, 128], strides: [1, 128] : memref<16384xi8> to memref<128x128xi8, strided<[1, 128]>>
        memref.copy %reinterpret_cast_5, %reinterpret_cast_7 {contract.access = {boundary = "external_ddr", kind = "load", mask = {kind = "none"}, origin_id = 2 : i64, other = 0 : i8}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 2>}, contract.task = {barrier_after = true, id = 9 : i64, input_values = array<i64: 1>, kind = "memory-load", origin_id = 2 : i64, output_values = array<i64: 15>, predecessors = array<i64>, rematerializable = false, scope = 4 : i64, stage = 0 : i64}} : memref<128x128xi8, strided<[1, 768], offset: ?>> to memref<128x128xi8, strided<[1, 128]>>
        linalg.matmul {contract.provenance = {a_origin_id = 1 : i64, a_source_lineage_complete = true, a_source_load_origin_ids = array<i64: 1>, accumulator_source_lineage_complete = true, accumulator_source_load_origin_ids = array<i64: 0>, b_origin_id = 2 : i64, b_source_lineage_complete = true, b_source_load_origin_ids = array<i64: 2>, origin_id = 3 : i64, source_lineage_complete = true, source_load_origin_ids = array<i64: 0, 1, 2>}, contract.task = {barrier_after = true, id = 10 : i64, input_values = array<i64: 14, 15, 11>, kind = "dot", output_values = array<i64: 16>, predecessors = array<i64: 6, 8, 9>, rematerializable = false, scope = 4 : i64, stage = 1 : i64}, cast = #linalg.type_fn<cast_signed>} ins(%alloc_2, %reinterpret_cast_7 : memref<32x128xi8>, memref<128x128xi8, strided<[1, 128]>>) outs(%alloc : memref<32x128xi32>)
      } {contract.cf = {kind = "uniform_for", serial = true, trip_count = 6 : i64, uniform = true}, contract.loop_carried_slots = [{backedge_producer_compute_ids = array<i64: 10>, exit_consumer_compute_ids = array<i64: 11>, init_producer_compute_ids = array<i64: 6>, init_value_id = 10 : i64, iter_arg_consumer_compute_ids = array<i64: 10>, iter_arg_value_id = 11 : i64, result_value_id = 12 : i64, slot = 0 : i64, yielded_value_id = 16 : i64}], contract.scope = {complete = true, id = 4 : i64, kind = "for-body", parent = 3 : i64}, contract.value_ids = array<i64: 11, 12>}
      memref.copy %subview_0, %subview {contract.access = {boundary = "external_ddr", kind = "store", mask = {kind = "rectangular", valid_shape = array<i64: 4, 128>}, origin_id = 4 : i64}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0, 1, 2>}, contract.task = {barrier_after = true, id = 11 : i64, input_values = array<i64: 12>, kind = "memory-store", origin_id = 4 : i64, output_values = array<i64>, predecessors = array<i64: 6, 10>, rematerializable = false, scope = 3 : i64, stage = 2 : i64}} : memref<4x128xi32, strided<[128, 1]>> to memref<4x128xi32, strided<[768, 1], offset: ?>>
    } {contract.cf = {arms_exclusive = true, kind = "uniform_if"}, contract.scopes = [{arm = "then", arms_exclusive = true, complete = true, id = 1 : i64, kind = "if-arm", parent = 0 : i64}, {arm = "else", arms_exclusive = true, complete = true, id = 3 : i64, kind = "if-arm", parent = 0 : i64}]}
    return
  }
}
