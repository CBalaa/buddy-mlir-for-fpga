// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t.first 2>%t.first.plan
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t.second 2>%t.second.plan
// RUN: cmp %t.first %t.second
// RUN: cmp %t.first.plan %t.second.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.first.plan

module attributes {bosc_ame.compute_graph_version = 1 : i32,
                   bosc_ame.dma.abi = "args_aligned64_nonalias"} {
  func.func @bounded(
      %arg0: memref<16xf32> {bosc_ame.logical_value_id = 100 : i64,
                             bosc_ame.logical_value_users = array<i64: 5, 6>},
      %arg1: memref<16xf32> {bosc_ame.logical_value_id = 101 : i64,
                             bosc_ame.logical_value_users = array<i64: 6, 7>},
      %arg2: memref<16xf32> {bosc_ame.logical_value_id = 102 : i64,
                             bosc_ame.logical_value_users = array<i64: 7, 8>},
      %arg3: memref<16xf32> {bosc_ame.logical_value_id = 103 : i64,
                             bosc_ame.logical_value_users = array<i64: 8, 9>},
      %seed0: memref<f32> {bosc_ame.logical_value_id = 104 : i64,
                           bosc_ame.logical_value_users = array<i64: 0>},
      %seed1: memref<f32> {bosc_ame.logical_value_id = 105 : i64,
                           bosc_ame.logical_value_users = array<i64: 1>},
      %seed2: memref<f32> {bosc_ame.logical_value_id = 106 : i64,
                           bosc_ame.logical_value_users = array<i64: 2>},
      %seed3: memref<f32> {bosc_ame.logical_value_id = 107 : i64,
                           bosc_ame.logical_value_users = array<i64: 3>},
      %seed4: memref<f32> {bosc_ame.logical_value_id = 108 : i64,
                           bosc_ame.logical_value_users = array<i64: 4>},
      %tmp0: memref<16xf32> {bosc_ame.logical_value_id = 109 : i64,
                             bosc_ame.logical_value_users = array<i64: 5>},
      %tmp1: memref<16xf32> {bosc_ame.logical_value_id = 110 : i64,
                             bosc_ame.logical_value_users = array<i64: 6>},
      %tmp2: memref<16xf32> {bosc_ame.logical_value_id = 111 : i64,
                             bosc_ame.logical_value_users = array<i64: 7>},
      %tmp3: memref<16xf32> {bosc_ame.logical_value_id = 112 : i64,
                             bosc_ame.logical_value_users = array<i64: 8>},
      %tmp4: memref<16xf32> {bosc_ame.logical_value_id = 113 : i64,
                             bosc_ame.logical_value_users = array<i64: 9>},
      %out0: memref<16xf32> {bosc_ame.logical_value_id = 114 : i64},
      %out1: memref<16xf32> {bosc_ame.logical_value_id = 115 : i64},
      %out2: memref<16xf32> {bosc_ame.logical_value_id = 116 : i64},
      %out3: memref<16xf32> {bosc_ame.logical_value_id = 117 : i64},
      %out4: memref<16xf32> {bosc_ame.logical_value_id = 118 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 10 : i64} {
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64: 104>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 109>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed0 : memref<f32>) outs(%tmp0 : memref<16xf32>) {
    ^bb0(%seed: f32, %old: f32):
      linalg.yield %seed : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 105>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 110>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed1 : memref<f32>) outs(%tmp1 : memref<16xf32>) {
    ^bb0(%seed: f32, %old: f32):
      linalg.yield %seed : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 2 : i64,
      bosc_ame.compute_input_values = array<i64: 106>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 111>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed2 : memref<f32>) outs(%tmp2 : memref<16xf32>) {
    ^bb0(%seed: f32, %old: f32):
      linalg.yield %seed : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 3 : i64,
      bosc_ame.compute_input_values = array<i64: 107>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 112>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed3 : memref<f32>) outs(%tmp3 : memref<16xf32>) {
    ^bb0(%seed: f32, %old: f32):
      linalg.yield %seed : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 4 : i64,
      bosc_ame.compute_input_values = array<i64: 108>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 113>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed4 : memref<f32>) outs(%tmp4 : memref<16xf32>) {
    ^bb0(%seed: f32, %old: f32):
      linalg.yield %seed : f32
    }

    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 5 : i64,
      bosc_ame.compute_input_values = array<i64: 109, 100>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 114>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp0, %arg0 : memref<16xf32>, memref<16xf32>)
      outs(%out0 : memref<16xf32>) {
    ^bb0(%value: f32, %other: f32, %old: f32):
      %sum = arith.addf %value, %other : f32
      linalg.yield %sum : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 6 : i64,
      bosc_ame.compute_input_values = array<i64: 110, 100, 101>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 115>,
      bosc_ame.compute_predecessors = array<i64: 1>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp1, %arg0, %arg1 : memref<16xf32>, memref<16xf32>,
          memref<16xf32>) outs(%out1 : memref<16xf32>) {
    ^bb0(%value: f32, %first: f32, %second: f32, %old: f32):
      %sum0 = arith.addf %value, %first : f32
      %sum1 = arith.addf %sum0, %second : f32
      linalg.yield %sum1 : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 7 : i64,
      bosc_ame.compute_input_values = array<i64: 111, 101, 102>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 116>,
      bosc_ame.compute_predecessors = array<i64: 2>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp2, %arg1, %arg2 : memref<16xf32>, memref<16xf32>,
          memref<16xf32>) outs(%out2 : memref<16xf32>) {
    ^bb0(%value: f32, %first: f32, %second: f32, %old: f32):
      %sum0 = arith.addf %value, %first : f32
      %sum1 = arith.addf %sum0, %second : f32
      linalg.yield %sum1 : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 8 : i64,
      bosc_ame.compute_input_values = array<i64: 112, 102, 103>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 117>,
      bosc_ame.compute_predecessors = array<i64: 3>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp3, %arg2, %arg3 : memref<16xf32>, memref<16xf32>,
          memref<16xf32>) outs(%out3 : memref<16xf32>) {
    ^bb0(%value: f32, %first: f32, %second: f32, %old: f32):
      %sum0 = arith.addf %value, %first : f32
      %sum1 = arith.addf %sum0, %second : f32
      linalg.yield %sum1 : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 9 : i64,
      bosc_ame.compute_input_values = array<i64: 113, 103>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 118>,
      bosc_ame.compute_predecessors = array<i64: 4>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp4, %arg3 : memref<16xf32>, memref<16xf32>)
      outs(%out4 : memref<16xf32>) {
    ^bb0(%value: f32, %other: f32, %old: f32):
      %sum = arith.addf %value, %other : f32
      linalg.yield %sum : f32
    }
    return
  }

  func.func @producer_candidates(
      %seed0: memref<f32> {bosc_ame.logical_value_id = 150 : i64,
                           bosc_ame.logical_value_users = array<i64: 10>},
      %seed1: memref<f32> {bosc_ame.logical_value_id = 151 : i64,
                           bosc_ame.logical_value_users = array<i64: 12>},
      %seed2: memref<f32> {bosc_ame.logical_value_id = 152 : i64,
                           bosc_ame.logical_value_users = array<i64: 14>},
      %seed3: memref<f32> {bosc_ame.logical_value_id = 153 : i64,
                           bosc_ame.logical_value_users = array<i64: 16>},
      %seed4: memref<f32> {bosc_ame.logical_value_id = 154 : i64,
                           bosc_ame.logical_value_users = array<i64: 18>},
      %tmp0: memref<16xf32> {bosc_ame.logical_value_id = 200 : i64,
                             bosc_ame.logical_value_users = array<i64: 11>},
      %tmp1: memref<16xf32> {bosc_ame.logical_value_id = 201 : i64,
                             bosc_ame.logical_value_users = array<i64: 13>},
      %tmp2: memref<16xf32> {bosc_ame.logical_value_id = 202 : i64,
                             bosc_ame.logical_value_users = array<i64: 15>},
      %tmp3: memref<16xf32> {bosc_ame.logical_value_id = 203 : i64,
                             bosc_ame.logical_value_users = array<i64: 17>},
      %tmp4: memref<16xf32> {bosc_ame.logical_value_id = 204 : i64,
                             bosc_ame.logical_value_users = array<i64: 19>},
      %out0: memref<16xf32> {bosc_ame.logical_value_id = 210 : i64},
      %out1: memref<16xf32> {bosc_ame.logical_value_id = 211 : i64},
      %out2: memref<16xf32> {bosc_ame.logical_value_id = 212 : i64},
      %out3: memref<16xf32> {bosc_ame.logical_value_id = 213 : i64},
      %out4: memref<16xf32> {bosc_ame.logical_value_id = 214 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 10 : i64} {
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 10 : i64,
      bosc_ame.compute_input_values = array<i64: 150>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 200>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed0 : memref<f32>) outs(%tmp0 : memref<16xf32>) {
    ^bb0(%seed: f32, %old: f32):
      linalg.yield %seed : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 11 : i64,
      bosc_ame.compute_input_values = array<i64: 200>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 210>,
      bosc_ame.compute_predecessors = array<i64: 10>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp0 : memref<16xf32>) outs(%out0 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 12 : i64,
      bosc_ame.compute_input_values = array<i64: 151>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 201>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed1 : memref<f32>) outs(%tmp1 : memref<16xf32>) {
    ^bb0(%seed: f32, %old: f32):
      linalg.yield %seed : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 13 : i64,
      bosc_ame.compute_input_values = array<i64: 201>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 211>,
      bosc_ame.compute_predecessors = array<i64: 12>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp1 : memref<16xf32>) outs(%out1 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 14 : i64,
      bosc_ame.compute_input_values = array<i64: 152>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 202>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed2 : memref<f32>) outs(%tmp2 : memref<16xf32>) {
    ^bb0(%seed: f32, %old: f32):
      linalg.yield %seed : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 15 : i64,
      bosc_ame.compute_input_values = array<i64: 202>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 212>,
      bosc_ame.compute_predecessors = array<i64: 14>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp2 : memref<16xf32>) outs(%out2 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 16 : i64,
      bosc_ame.compute_input_values = array<i64: 153>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 203>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed3 : memref<f32>) outs(%tmp3 : memref<16xf32>) {
    ^bb0(%seed: f32, %old: f32):
      linalg.yield %seed : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 17 : i64,
      bosc_ame.compute_input_values = array<i64: 203>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 213>,
      bosc_ame.compute_predecessors = array<i64: 16>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp3 : memref<16xf32>) outs(%out3 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 18 : i64,
      bosc_ame.compute_input_values = array<i64: 154>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 204>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> ()>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%seed4 : memref<f32>) outs(%tmp4 : memref<16xf32>) {
    ^bb0(%seed: f32, %old: f32):
      linalg.yield %seed : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 19 : i64,
      bosc_ame.compute_input_values = array<i64: 204>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 214>,
      bosc_ame.compute_predecessors = array<i64: 18>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp4 : memref<16xf32>) outs(%out4 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// The report summarizes the complete module: @bounded contributes four
// argument and five producer candidates; @producer_candidates adds five more
// producer candidates, for fourteen joint candidates and ten producers.
// PLAN: function-argument-residency-search: candidates=4 selected=4 rejected-capacity=0 objective=minimize-read-bytes-then-packets status=bounded latency=uncalibrated
// PLAN: cross-task-residency-search: candidates=14 selected=14 rejected-capacity=0 objective=minimize-total-read-bytes-then-packets status=bounded latency=uncalibrated
// PLAN: producer-output-residency-search: candidates=10 selected=10 rejected-capacity=0 objective=maximize-read-bytes-then-packets-saved status=bounded latency=uncalibrated
