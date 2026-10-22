// RUN: split-file %s %t
// RUN: buddy-opt %t/reader.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t/reader.out 2>%t/reader.plan
// RUN: FileCheck %s --check-prefix=UNSAFE-PLAN < %t/reader.plan
// RUN: FileCheck %s --check-prefix=READER-IR < %t/reader.out
// RUN: buddy-opt %t/produced.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t/produced.out 2>%t/produced.plan
// RUN: FileCheck %s --check-prefix=UNSAFE-PLAN < %t/produced.plan
// RUN: FileCheck %s --check-prefix=PRODUCED-IR < %t/produced.out
// RUN: buddy-opt %t/entry.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t/entry.out 2>%t/entry.plan
// RUN: FileCheck %s --check-prefix=ENTRY-PLAN < %t/entry.plan
// RUN: buddy-opt %t/fresh.mlir --linalg-to-boscame='enable-spm-dma=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t/fresh.out 2>%t/fresh.plan
// RUN: FileCheck %s --check-prefix=FRESH-PLAN < %t/fresh.plan

// The module ABI applies to external entry arguments. It cannot make distinct
// helper formals disjoint when a caller passes the same actual buffer twice.
// The reader helper is public: visibility alone does not make it an entry.
// Both unsafe cases must retain their DDR writes and subsequent DDR reads.
// UNSAFE-PLAN-NOT: residency=cross-task
// UNSAFE-PLAN: resources: transfers=4 dma-packets=4 read-bytes=128 write-bytes=128 totals-exact=true latency=uncalibrated
// UNSAFE-PLAN: function-argument-residency-search: candidates=0 selected=0

// The first consumer changes the aliased source by adding one. The second
// consumer must reload that updated source after the first output write.
// READER-IR-LABEL: func.func @reader_helper(
// READER-IR-SAME: %[[SRC:arg[0-9]+]]: memref<16xf32>
// READER-IR-SAME: %[[ALIAS:arg[0-9]+]]: memref<16xf32>
// READER-IR-SAME: %[[DST:arg[0-9]+]]: memref<16xf32>
// READER-IR: %[[READ0:.*]] = memref.reinterpret_cast %[[SRC]] to
// READER-IR: bosc_ame.dma.start %[[READ0]],
// READER-IR: arith.addf
// READER-IR: %[[WRITE0:.*]] = memref.reinterpret_cast %[[ALIAS]] to
// READER-IR: bosc_ame.dma.start {{.*}}, %[[WRITE0]],
// READER-IR: bosc_ame.dma.wait
// READER-IR: %[[READ1:.*]] = memref.reinterpret_cast %[[SRC]] to
// READER-IR: bosc_ame.dma.start %[[READ1]],
// READER-IR: %[[WRITE1:.*]] = memref.reinterpret_cast %[[DST]] to
// READER-IR: bosc_ame.dma.start {{.*}}, %[[WRITE1]],
// READER-IR: return
// READER-IR-LABEL: func.func @reader_entry(
// READER-IR: call @reader_helper(%[[ACTUAL:arg[0-9]+]], %[[ACTUAL]],

// The producer writes two to %tmp, then the consumer adds %tmp and the
// aliasing formal %other. Its result must be four even when %other held a
// different value before the call. The producer write must precede both reads.
// PRODUCED-IR-LABEL: func.func private @produced_helper(
// PRODUCED-IR-SAME: %[[TMP:arg[0-9]+]]: memref<16xf32>
// PRODUCED-IR-SAME: %[[OTHER:arg[0-9]+]]: memref<16xf32>
// PRODUCED-IR-SAME: %[[OUT:arg[0-9]+]]: memref<16xf32>
// PRODUCED-IR: %[[TMP_WRITE:.*]] = memref.reinterpret_cast %[[TMP]] to
// PRODUCED-IR: bosc_ame.dma.start {{.*}}, %[[TMP_WRITE]],
// PRODUCED-IR: bosc_ame.dma.wait
// PRODUCED-IR: %[[TMP_READ:.*]] = memref.reinterpret_cast %[[TMP]] to
// PRODUCED-IR: bosc_ame.dma.start %[[TMP_READ]],
// PRODUCED-IR: %[[OTHER_READ:.*]] = memref.reinterpret_cast %[[OTHER]] to
// PRODUCED-IR: bosc_ame.dma.start %[[OTHER_READ]],
// PRODUCED-IR: arith.addf
// PRODUCED-IR: %[[OUT_WRITE:.*]] = memref.reinterpret_cast %[[OUT]] to
// PRODUCED-IR: bosc_ame.dma.start {{.*}}, %[[OUT_WRITE]],
// PRODUCED-IR: return
// PRODUCED-IR-LABEL: func.func @produced_entry(
// PRODUCED-IR: call @produced_helper(%[[ACTUAL:arg[0-9]+]], %[[ACTUAL]],

// An uncalled public entry retains the ABI proof for its distinct arguments.
// ENTRY-PLAN: transfer[0]: consumer=linalg.generic residency=cross-task consumers=[linalg.generic, linalg.generic] role=input-a
// ENTRY-PLAN: resources: transfers=3 dma-packets=3 read-bytes=64 write-bytes=128 totals-exact=true latency=uncalibrated
// ENTRY-PLAN: function-argument-residency-search: candidates=1 selected=1

// Fresh local storage remains disjoint from helper arguments even if those
// arguments themselves alias. Its producer output can stay resident safely.
// FRESH-PLAN: transfer[0]: consumer=linalg.generic residency=cross-task consumers=[linalg.generic, linalg.generic, linalg.generic] role=output-c
// FRESH-PLAN: resources: transfers=3 dma-packets=3 read-bytes=0 write-bytes=192 totals-exact=true latency=uncalibrated

//--- reader.mlir
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @reader_helper(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 0, 1>},
      %alias: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64},
      %dst: memref<16xf32> {bosc_ame.logical_value_id = 12 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
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
      ins(%src : memref<16xf32>) outs(%alias : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      %one = arith.constant 1.0 : f32
      %incremented = arith.addf %value, %one : f32
      linalg.yield %incremented : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 12>,
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
    return
  }

  func.func @reader_entry(%src: memref<16xf32>, %dst: memref<16xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 0 : i64} {
    func.call @reader_helper(%src, %src, %dst)
      : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
    return
  }
}

//--- produced.mlir
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func private @produced_helper(
      %tmp: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 1>},
      %other: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64,
                               bosc_ame.logical_value_users = array<i64: 1>},
      %dst: memref<16xf32> {bosc_ame.logical_value_id = 12 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 10>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      outs(%tmp : memref<16xf32>) {
    ^bb0(%old: f32):
      %two = arith.constant 2.0 : f32
      linalg.yield %two : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 10, 11>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp, %other : memref<16xf32>, memref<16xf32>)
      outs(%dst : memref<16xf32>) {
    ^bb0(%value: f32, %alias: f32, %old: f32):
      %sum = arith.addf %value, %alias : f32
      linalg.yield %sum : f32
    }
    return
  }

  func.func @produced_entry(%tmp: memref<16xf32>, %dst: memref<16xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 0 : i64} {
    func.call @produced_helper(%tmp, %tmp, %dst)
      : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()
    return
  }
}

//--- entry.mlir
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func @public_entry(
      %src: memref<16xf32> {bosc_ame.logical_value_id = 10 : i64,
                             bosc_ame.logical_value_users = array<i64: 0, 1>},
      %dst0: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64},
      %dst1: memref<16xf32> {bosc_ame.logical_value_id = 12 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 2 : i64} {
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
      ins(%src : memref<16xf32>) outs(%dst0 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%src : memref<16xf32>) outs(%dst1 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

//--- fresh.mlir
module attributes {bosc_ame.compute_graph_version = 1 : i32} {
  func.func private @fresh_helper(
      %dst0: memref<16xf32> {bosc_ame.logical_value_id = 11 : i64},
      %dst1: memref<16xf32> {bosc_ame.logical_value_id = 12 : i64})
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 3 : i64} {
    %tmp = memref.alloc() alignment = 64 : memref<16xf32>
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 0 : i64,
      bosc_ame.compute_input_values = array<i64>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 10>,
      bosc_ame.compute_predecessors = array<i64>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      outs(%tmp : memref<16xf32>) {
    ^bb0(%old: f32):
      %two = arith.constant 2.0 : f32
      linalg.yield %two : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 1 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 11>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp : memref<16xf32>) outs(%dst0 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    linalg.generic {
      bosc_ame.compute_barrier_after = false,
      bosc_ame.compute_id = 2 : i64,
      bosc_ame.compute_input_values = array<i64: 10>,
      bosc_ame.compute_kind = "elementwise",
      bosc_ame.compute_output_values = array<i64: 12>,
      bosc_ame.compute_predecessors = array<i64: 0>,
      bosc_ame.compute_stage = 0 : i64,
      bosc_ame.rematerializable = true,
      bosc_ame.varying_program_axes = array<i64>,
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%tmp : memref<16xf32>) outs(%dst1 : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }

  func.func @fresh_entry(%dst: memref<16xf32>)
      attributes {bosc_ame.compute_graph_complete = true,
                  bosc_ame.compute_graph_node_count = 0 : i64} {
    func.call @fresh_helper(%dst, %dst)
      : (memref<16xf32>, memref<16xf32>) -> ()
    return
  }
}
