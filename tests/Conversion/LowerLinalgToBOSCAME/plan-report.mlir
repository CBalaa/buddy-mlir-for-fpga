// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' -o %t.default 2>%t.default.stderr
// RUN: test ! -s %t.default.stderr
// RUN: buddy-opt %s --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga plan-report=true' -o %t.reported 2>%t.plan
// RUN: cmp %t.default %t.reported
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan

module {
  func.func @gemm(%numPrograms0: i32 {bosc_ame.program_axis = 0 : i32, bosc_ame.program_role = "num_programs"},
                  %numPrograms1: i32 {bosc_ame.program_axis = 1 : i32, bosc_ame.program_role = "num_programs"},
                  %numPrograms2: i32 {bosc_ame.program_axis = 2 : i32, bosc_ame.program_role = "num_programs"},
                  %a: memref<64x64xi8>,
                  %b: memref<64x64xi8, strided<[1, 64]>>,
                  %c: memref<64x64xi32>) {
    linalg.matmul {bosc_ame.a_origin_id = 101 : i64,
                   bosc_ame.b_origin_id = 202 : i64,
                   cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>,
                   memref<64x64xi8, strided<[1, 64]>>)
      outs(%c : memref<64x64xi32>)
    return
  }

  func.func @copy(%input: memref<16xf32>, %output: memref<16xf32>) {
    linalg.generic {
      bosc_ame.memory = {destination = "external_ddr", kind = "store",
                         logical_shape = array<i64: 16>, masked = false,
                         origin_id = 23 : i64, valid_shape = array<i64: 16>},
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%input : memref<16xf32>) outs(%output : memref<16xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}

// PLAN: BOSCAME plan report:
// PLAN-NEXT:   target=nr-fpga tile=[64,64,128] routing=automatic bank-capacity=131072 bytes
// PLAN-NEXT:   target-capabilities: name=nr-fpga dma={ddr-to-spm=true,spm-to-ddr=true,spm-to-spm=true,endpoint-alignment=64,packet-bytes=64,max-rank=2,pitched-rows=packetized} banks=2 channels=2 ame={signed-multiply=true,tile-mnk=64x64x64,a={bits=8,layout=mk-k-contiguous,spm=true,direct-ddr=true},b={bits=8,layout=kn-physical-nk-k-contiguous,spm=true,direct-ddr=true},accumulator={bits=32,layout=mn-n-contiguous,spm=true,direct-ddr=true}} rvv-vector-bytes=<unknown>
// PLAN:   launch-grid: function=@gemm axis[0]=runtime(arg#0) extent=<unknown> axis[1]=runtime(arg#1) extent=<unknown> axis[2]=runtime(arg#2) extent=<unknown>
// PLAN:   launch-grid: function=@copy unavailable
// PLAN:   transfer[0]: consumer=linalg.matmul residency=operation-local consumers=[linalg.matmul] source-origin-id=101 role=input-a shape=[64,64] strides=[64,1] validShape=[64,64] bank=0 offset=0 capacity=4096 effective-byte-offset-bytes=0 endpoint-alignment-proven=true packets=1 read=true write=false
// PLAN:   transfer[1]: consumer=linalg.matmul residency=operation-local consumers=[linalg.matmul] source-origin-id=202 role=input-b shape=[64,64] strides=[1,64] validShape=[64,64] bank=1 offset=0 capacity=4096 effective-byte-offset-bytes=0 endpoint-alignment-proven=true packets=1 read=true write=false
// PLAN:   transfer[2]: consumer=linalg.generic residency=operation-local consumers=[linalg.generic] source-origin-id=23 role=input-a shape=[16] strides=[1] validShape=[16] bank=0 offset=0 capacity=64 effective-byte-offset-bytes=0 endpoint-alignment-proven=true packets=1 read=true write=false
// PLAN:   compute-graph: status=unavailable tasks=0 dependencies=0 logical-values=0
// PLAN:   resources: transfers=3 dma-packets=3 read-bytes=8256 write-bytes=0 totals-exact=true latency=uncalibrated
// PLAN:   bank[0]: high-water=4096/131072 bytes
// PLAN:   bank[1]: high-water=4096/131072 bytes
// PLAN:   marked-matmul: execution=AME m=64 n=64 k=64 a_origin_id=101 b_origin_id=202 accumulator_origin_id=<unknown> direct-A=spm-dma (planned) direct-B=spm-dma (planned) direct-C=direct-ddr (proven)
// PLAN:   provenance-marked stores/direct decisions:
// PLAN: op=linalg.generic destination=external_ddr decision=direct-ddr (proven)
// PLAN:   limits: calibrated cost search, general cross-op residency, and cross-program reuse are not modeled.
