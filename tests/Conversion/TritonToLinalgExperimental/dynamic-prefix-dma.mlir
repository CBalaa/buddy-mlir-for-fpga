// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' --empty-tensor-to-alloc-tensor --one-shot-bufferize='bufferize-function-boundaries=true buffer-alignment=64' --canonicalize --cse --linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t.ir 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: FileCheck %s --check-prefix=IR --implicit-check-not=memref.copy < %t.ir

module {
  tt.func public @dynamic_prefix_aligned(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %limit: i32) {
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %packet_mask = arith.constant -16 : i32
    %aligned_bound = arith.andi %limit, %packet_mask : i32
    %bound = tt.splat %aligned_bound : i32 -> tensor<32xi32>
    %mask = arith.cmpi slt, %range, %bound : tensor<32xi32>
    %other = arith.constant dense<0.0> : tensor<32xf32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %src_ptr = tt.addptr %srcs, %range : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %dst_ptr = tt.addptr %dsts, %range : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %value = tt.load %src_ptr, %mask, %other : tensor<32x!tt.ptr<f32>>
    tt.store %dst_ptr, %value, %mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// PLAN: guarded-prefix=true packet-count=upper-bound extent=
// PLAN: resources: {{.*}}totals-exact=false

// A load packet's guard encloses its immediate wait and the SPM-to-local copy.
// IR: [[ZERO:%.*]] = vector.broadcast %cst : f32 to vector<16xf32>
// IR: scf.for {{.*}} to %c32{{(_[0-9]+)?}} step %c16{{(_[0-9]+)?}}
// IR: vector.store [[ZERO]]
// IR: arith.cmpi sge, {{.*}}, %c16{{(_[0-9]+)?}} : index
// IR: scf.if
// IR: bosc_ame.dma.start {{.*}}, %c64
// IR-NEXT: bosc_ame.dma.wait
// IR: vector.load
// IR: vector.store
// IR: arith.cmpi sge, {{.*}}, %c32{{(_[0-9]+)?}} : index
// IR: scf.if
// IR: bosc_ame.dma.start {{.*}}, %c64
// IR-NEXT: bosc_ame.dma.wait

// A store packet is guarded through both its source read and DDR write.
// IR: arith.cmpi sge, {{.*}}, %c16{{(_[0-9]+)?}} : index
// IR: scf.if
// IR: bosc_ame.dma.start {{.*}}, %c64
// IR-NEXT: bosc_ame.dma.wait
// IR: bosc_ame.dma.start {{.*}}, %c64
// IR-NEXT: bosc_ame.dma.wait
// IR: arith.cmpi sge, {{.*}}, %c32{{(_[0-9]+)?}} : index
// IR: scf.if
// IR: bosc_ame.dma.start {{.*}}, %c64
// IR-NEXT: bosc_ame.dma.wait
// IR: bosc_ame.dma.start {{.*}}, %c64
// IR-NEXT: bosc_ame.dma.wait
