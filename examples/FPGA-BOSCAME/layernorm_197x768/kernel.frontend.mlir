#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}, contract.producer = "triton-riscv", contract.version = 1 : i32} {
  func.func @triton_layernorm_197x768(%arg0: memref<*xf32> {contract.value_id = 0 : i64}, %arg1: memref<*xf32> {contract.value_id = 1 : i64}, %arg2: memref<*xf32> {contract.value_id = 2 : i64}, %arg3: memref<*xf32> {contract.value_id = 3 : i64}, %arg4: i32 {contract.program = {axis = 0 : i32, role = "num_programs"}}, %arg5: i32 {contract.program = {axis = 1 : i32, role = "num_programs"}}, %arg6: i32 {contract.program = {axis = 2 : i32, role = "num_programs"}}, %arg7: i32 {contract.program = {axis = 0 : i32, role = "program_id"}}, %arg8: i32 {contract.program = {axis = 1 : i32, role = "program_id"}}, %arg9: i32 {contract.program = {axis = 2 : i32, role = "program_id"}}) attributes {contract.compute_graph = {complete = true, root_scope = 0 : i64, scope_count = 1 : i64, source_lineage_complete = true, version = 1 : i32}, contract.grid = array<i32: 197, 1, 1>, gpu.known_grid_size = array<i32: 197, 1, 1>, llvm.emit_c_interface, noinline = false} {
    %cst = arith.constant 7.680000e+02 : f32
    %cst_0 = arith.constant 9.99999996E-13 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c768_i32 = arith.constant 768 : i32
    %0 = arith.muli %arg7, %c768_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 1024>, role = "tile"}} : memref<1024xf32>
    linalg.fill {contract.fill = {padding = true}, contract.task = {barrier_after = false, id = 0 : i64, input_values = array<i64>, kind = "fill", output_values = array<i64: 4>, predecessors = array<i64>, rematerializable = false, scope = 0 : i64, stage = 0 : i64}} ins(%cst_1 : f32) outs(%alloc : memref<1024xf32>)
    %subview = memref.subview %reinterpret_cast[0] [768] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    %subview_3 = memref.subview %alloc[0] [768] [1] : memref<1024xf32> to memref<768xf32, strided<[1]>>
    memref.copy %subview, %subview_3 {contract.access = {boundary = "external_ddr", kind = "load", mask = {kind = "rectangular", valid_shape = array<i64: 768>}, origin_id = 0 : i64, other = 0.000000e+00 : f32}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = true, id = 1 : i64, input_values = array<i64: 0, 4>, kind = "memory-load", origin_id = 0 : i64, output_values = array<i64: 5>, predecessors = array<i64: 0>, rematerializable = false, scope = 0 : i64, stage = 0 : i64}} : memref<768xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1]>>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
    %alloc_5 = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 1024>, role = "tile"}} : memref<1024xf32>
    linalg.fill {contract.fill = {padding = true}, contract.task = {barrier_after = false, id = 2 : i64, input_values = array<i64>, kind = "fill", output_values = array<i64: 6>, predecessors = array<i64>, rematerializable = false, scope = 0 : i64, stage = 0 : i64}} ins(%cst_2 : f32) outs(%alloc_5 : memref<1024xf32>)
    %subview_6 = memref.subview %reinterpret_cast_4[0] [768] [1] : memref<1024xf32, strided<[1]>> to memref<768xf32, strided<[1]>>
    %subview_7 = memref.subview %alloc_5[0] [768] [1] : memref<1024xf32> to memref<768xf32, strided<[1]>>
    memref.copy %subview_6, %subview_7 {contract.access = {boundary = "external_ddr", kind = "load", mask = {kind = "rectangular", valid_shape = array<i64: 768>}, origin_id = 1 : i64, other = 1.000000e+00 : f32}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 1>}, contract.task = {barrier_after = true, id = 3 : i64, input_values = array<i64: 1, 6>, kind = "memory-load", origin_id = 1 : i64, output_values = array<i64: 7>, predecessors = array<i64: 2>, rematerializable = false, scope = 0 : i64, stage = 0 : i64}} : memref<768xf32, strided<[1]>> to memref<768xf32, strided<[1]>>
    %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
    %alloc_9 = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 1024>, role = "tile"}} : memref<1024xf32>
    linalg.fill {contract.fill = {padding = true}, contract.task = {barrier_after = false, id = 4 : i64, input_values = array<i64>, kind = "fill", output_values = array<i64: 8>, predecessors = array<i64>, rematerializable = false, scope = 0 : i64, stage = 0 : i64}} ins(%cst_1 : f32) outs(%alloc_9 : memref<1024xf32>)
    %subview_10 = memref.subview %reinterpret_cast_8[0] [768] [1] : memref<1024xf32, strided<[1]>> to memref<768xf32, strided<[1]>>
    %subview_11 = memref.subview %alloc_9[0] [768] [1] : memref<1024xf32> to memref<768xf32, strided<[1]>>
    memref.copy %subview_10, %subview_11 {contract.access = {boundary = "external_ddr", kind = "load", mask = {kind = "rectangular", valid_shape = array<i64: 768>}, origin_id = 2 : i64, other = 0.000000e+00 : f32}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 2>}, contract.task = {barrier_after = true, id = 5 : i64, input_values = array<i64: 2, 8>, kind = "memory-load", origin_id = 2 : i64, output_values = array<i64: 9>, predecessors = array<i64: 4>, rematerializable = false, scope = 0 : i64, stage = 0 : i64}} : memref<768xf32, strided<[1]>> to memref<768xf32, strided<[1]>>
    %alloc_12 = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64>, role = "scratch"}} : memref<f32>
    linalg.fill {contract.task = {barrier_after = false, id = 6 : i64, input_values = array<i64>, kind = "fill", output_values = array<i64: 10>, predecessors = array<i64>, rematerializable = false, scope = 0 : i64, stage = 0 : i64}} ins(%cst_1 : f32) outs(%alloc_12 : memref<f32>)
    %alloc_13 = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64>, role = "scratch"}} : memref<f32>
    linalg.fill {contract.task = {barrier_after = false, id = 7 : i64, input_values = array<i64>, kind = "fill", output_values = array<i64: 11>, predecessors = array<i64>, rematerializable = false, scope = 0 : i64, stage = 0 : i64}} ins(%cst_1 : f32) outs(%alloc_13 : memref<f32>)
    linalg.reduce ins(%alloc : memref<1024xf32>) outs(%alloc_13 : memref<f32>) dimensions = [0]  {contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = true, id = 8 : i64, input_values = array<i64: 5, 11>, kind = "reduce", output_values = array<i64: 12>, predecessors = array<i64: 1, 7>, rematerializable = false, scope = 0 : i64, stage = 1 : i64}}
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    %2 = memref.load %alloc_13[] {contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = false, id = 9 : i64, input_values = array<i64: 12>, kind = "scalar-load", output_values = array<i64>, predecessors = array<i64: 8>, rematerializable = false, scope = 0 : i64, stage = 2 : i64}} : memref<f32>
    %3 = arith.divf %2, %cst : f32
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc, %3 : memref<1024xf32>, f32) outs(%alloc : memref<1024xf32>) attrs =  {contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = false, id = 10 : i64, input_values = array<i64: 5>, kind = "elementwise", output_values = array<i64: 13>, predecessors = array<i64: 1, 8, 9>, rematerializable = false, scope = 0 : i64, stage = 2 : i64}} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %8 = arith.subf %in, %in_20 : f32
      linalg.yield %8 : f32
    }
    %subview_14 = memref.subview %alloc[768] [256] [1] : memref<1024xf32> to memref<256xf32, strided<[1], offset: 768>>
    linalg.fill {contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = false, id = 11 : i64, input_values = array<i64: 13>, kind = "fill", output_values = array<i64: 14>, predecessors = array<i64: 10>, rematerializable = false, scope = 0 : i64, stage = 2 : i64}} ins(%cst_1 : f32) outs(%subview_14 : memref<256xf32, strided<[1], offset: 768>>)
    %alloc_15 = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 1024>, role = "tile"}} : memref<1024xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc : memref<1024xf32>) outs(%alloc_15 : memref<1024xf32>) attrs =  {contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = false, id = 12 : i64, input_values = array<i64: 14>, kind = "elementwise", output_values = array<i64: 15>, predecessors = array<i64: 11>, rematerializable = false, scope = 0 : i64, stage = 2 : i64}} {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %in : f32
      linalg.yield %8 : f32
    }
    linalg.reduce ins(%alloc_15 : memref<1024xf32>) outs(%alloc_12 : memref<f32>) dimensions = [0]  {contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = true, id = 13 : i64, input_values = array<i64: 15, 10>, kind = "reduce", output_values = array<i64: 16>, predecessors = array<i64: 6, 12>, rematerializable = false, scope = 0 : i64, stage = 2 : i64}}
      (%in: f32, %init: f32) {
        %8 = arith.addf %in, %init : f32
        linalg.yield %8 : f32
      }
    %4 = memref.load %alloc_12[] {contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = false, id = 14 : i64, input_values = array<i64: 16>, kind = "scalar-load", output_values = array<i64>, predecessors = array<i64: 13>, rematerializable = false, scope = 0 : i64, stage = 3 : i64}} : memref<f32>
    %5 = arith.divf %4, %cst : f32
    %6 = arith.addf %5, %cst_0 : f32
    %7 = math.rsqrt %6 : f32
    %alloc_16 = memref.alloc() alignment = 64 {contract.buffer = {kind = "temporary", logical_shape = array<i64: 1024>, role = "tile"}} : memref<1024xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc, %7 : memref<1024xf32>, f32) outs(%alloc_16 : memref<1024xf32>) attrs =  {contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0>}, contract.task = {barrier_after = false, id = 15 : i64, input_values = array<i64: 14>, kind = "elementwise", output_values = array<i64: 17>, predecessors = array<i64: 11, 14>, rematerializable = false, scope = 0 : i64, stage = 3 : i64}} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %8 = arith.mulf %in, %in_20 : f32
      linalg.yield %8 : f32
    }
    linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc, %alloc_5, %7 : memref<1024xf32>, memref<1024xf32>, f32) outs(%alloc_16 : memref<1024xf32>) attrs =  {contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0, 1>}, contract.task = {barrier_after = false, id = 16 : i64, input_values = array<i64: 14, 7>, kind = "elementwise", output_values = array<i64: 18>, predecessors = array<i64: 3, 11, 14, 15>, rematerializable = false, scope = 0 : i64, stage = 3 : i64}} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %out: f32):
      %8 = arith.mulf %in, %in_21 : f32
      %9 = arith.mulf %8, %in_20 : f32
      linalg.yield %9 : f32
    }
    linalg.generic {indexing_maps = [#map, #map, #map, #map1, #map], iterator_types = ["parallel"]} ins(%alloc, %alloc_5, %alloc_9, %7 : memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, f32) outs(%alloc_16 : memref<1024xf32>) attrs =  {contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0, 1, 2>}, contract.task = {barrier_after = false, id = 17 : i64, input_values = array<i64: 14, 7, 9>, kind = "elementwise", output_values = array<i64: 19>, predecessors = array<i64: 3, 5, 11, 14, 16>, rematerializable = false, scope = 0 : i64, stage = 3 : i64}} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %out: f32):
      %8 = arith.mulf %in, %in_22 : f32
      %9 = arith.mulf %8, %in_20 : f32
      %10 = arith.addf %9, %in_21 : f32
      linalg.yield %10 : f32
    }
    %reinterpret_cast_17 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %subview_18 = memref.subview %alloc_16[0] [768] [1] : memref<1024xf32> to memref<768xf32, strided<[1]>>
    %subview_19 = memref.subview %reinterpret_cast_17[0] [768] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<768xf32, strided<[1], offset: ?>>
    memref.copy %subview_18, %subview_19 {contract.access = {boundary = "external_ddr", kind = "store", mask = {kind = "rectangular", valid_shape = array<i64: 768>}, origin_id = 3 : i64}, contract.provenance = {source_lineage_complete = true, source_load_origin_ids = array<i64: 0, 1, 2>}, contract.task = {barrier_after = true, id = 18 : i64, input_values = array<i64: 19>, kind = "memory-store", origin_id = 3 : i64, output_values = array<i64>, predecessors = array<i64: 17>, rematerializable = false, scope = 0 : i64, stage = 3 : i64}} : memref<768xf32, strided<[1]>> to memref<768xf32, strided<[1], offset: ?>>
    return
  }
}
