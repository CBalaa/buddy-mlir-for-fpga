// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @yielded_memref_alias(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                            bosc_ame.logical_value_users = array<i64: 0>},
      %dst: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64},
      %condition: i1)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 1 : i64} {
    %alias = scf.if %condition -> (memref<16xf32>) {
      scf.yield %src : memref<16xf32>
    } else {
      scf.yield %src : memref<16xf32>
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 11>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    %zero = arith.constant 0 : index
    %value = memref.load %alias[%zero] : memref<16xf32>
    return
  }

  func.func @iterated_memref_alias(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 20 : i64,
                            bosc_ame.logical_value_users = array<i64: 1>},
      %dst: memref<16xf32> {bosc_ame.logical_value_id = 21 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 1 : i64} {
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 20>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 21>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%src : memref<16xf32>) outs(%dst : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %alias = scf.for %iv = %zero to %one step %one
        iter_args(%iter = %src) -> (memref<16xf32>) {
      scf.yield %iter : memref<16xf32>
    }
    %value = memref.load %alias[%zero] : memref<16xf32>
    return
  }

  func.func @divergent_memref_alias(
      %src: memref<16xf32>, %other: memref<16xf32>, %condition: i1) {
    %alias = scf.if %condition -> (memref<16xf32>) {
      scf.yield %src : memref<16xf32>
    } else {
      scf.yield %other : memref<16xf32>
    }
    %zero = arith.constant 0 : index
    %value = memref.load %alias[%zero] : memref<16xf32>
    return
  }

  func.func @changed_view_memref_alias(
      %src: memref<32xf32>, %condition: i1) {
    %zero = arith.constant 0 : index
    %middle = arith.constant 16 : index
    %first = memref.subview %src[%zero] [16] [1]
        : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
    %second = memref.subview %src[%middle] [16] [1]
        : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
    %alias = scf.if %condition -> (memref<16xf32, strided<[1], offset: ?>>) {
      scf.yield %first : memref<16xf32, strided<[1], offset: ?>>
    } else {
      scf.yield %second : memref<16xf32, strided<[1], offset: ?>>
    }
    %value = memref.load %alias[%zero]
        : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @forwarded_view_intervals(
      %src: memref<32xf32>, %condition: i1) {
    %zero = arith.constant 0 : index
    %middle = arith.constant 16 : index
    %middle_view = memref.subview %src[%middle] [16] [1]
        : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
    %alias = scf.if %condition -> (memref<16xf32, strided<[1], offset: ?>>) {
      scf.yield %middle_view : memref<16xf32, strided<[1], offset: ?>>
    } else {
      scf.yield %middle_view : memref<16xf32, strided<[1], offset: ?>>
    }
    %forwarded = memref.subview %alias[%zero] [16] [1]
        : memref<16xf32, strided<[1], offset: ?>>
          to memref<16xf32, strided<[1], offset: ?>>
    %first_half = memref.subview %src[%zero] [16] [1]
        : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
    %first = memref.load %first_half[%zero]
        : memref<16xf32, strided<[1], offset: ?>>
    %second = memref.load %forwarded[%zero]
        : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @while_memref_alias(
      %src: memref<16xf32>, %condition: i1) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %alias = scf.while (%before = %src) : (memref<16xf32>) -> memref<16xf32> {
      scf.condition(%condition) %before : memref<16xf32>
    } do {
    ^bb0(%after: memref<16xf32>):
      scf.yield %after : memref<16xf32>
    }
    %value = memref.load %alias[%zero] : memref<16xf32>
    return
  }

  func.func @while_changed_view_alias(
      %src: memref<32xf32>, %condition: i1) {
    %zero = arith.constant 0 : index
    %middle = arith.constant 16 : index
    %alias = scf.while (%before = %src)
        : (memref<32xf32>) -> memref<16xf32, strided<[1], offset: ?>> {
      %view = memref.subview %before[%middle] [16] [1]
          : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
      scf.condition(%condition) %view : memref<16xf32, strided<[1], offset: ?>>
    } do {
    ^bb0(%after: memref<16xf32, strided<[1], offset: ?>>):
      scf.yield %src : memref<32xf32>
    }
    %value = memref.load %alias[%zero]
        : memref<16xf32, strided<[1], offset: ?>>
    return
  }

  func.func @index_switch_memref_alias(
      %src: memref<16xf32>, %selector: index) {
    %alias = scf.index_switch %selector -> memref<16xf32>
    case 0 {
      scf.yield %src : memref<16xf32>
    }
    default {
      scf.yield %src : memref<16xf32>
    }
    %zero = arith.constant 0 : index
    %value = memref.load %alias[%zero] : memref<16xf32>
    return
  }

  func.func @index_switch_divergent_alias(
      %src: memref<16xf32>, %other: memref<16xf32>, %selector: index) {
    %alias = scf.index_switch %selector -> memref<16xf32>
    case 0 {
      scf.yield %src : memref<16xf32>
    }
    default {
      scf.yield %other : memref<16xf32>
    }
    %zero = arith.constant 0 : index
    %value = memref.load %alias[%zero] : memref<16xf32>
    return
  }

  func.func @select_memref_alias(
      %src: memref<16xf32>, %condition: i1) {
    %alias = arith.select %condition, %src, %src : memref<16xf32>
    %zero = arith.constant 0 : index
    %value = memref.load %alias[%zero] : memref<16xf32>
    return
  }

  func.func @select_divergent_alias(
      %src: memref<16xf32>, %other: memref<16xf32>, %condition: i1) {
    %alias = arith.select %condition, %src, %other : memref<16xf32>
    %zero = arith.constant 0 : index
    %value = memref.load %alias[%zero] : memref<16xf32>
    return
  }
}
