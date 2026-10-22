// RUN: buddy-opt %s --triton-to-linalg-experimental='boscame=true dump-facts=true dma-abi=args_aligned64_nonalias' -o %t.linalg 2> %t.facts
// RUN: FileCheck %s --check-prefix=FACTS < %t.facts
// RUN: FileCheck %s --check-prefix=IR < %t.linalg
// RUN: not buddy-opt %s --triton-to-linalg-experimental='boscame=true dma-abi=args_aligned64_nonalias' --empty-tensor-to-alloc-tensor --one-shot-bufferize='bufferize-function-boundaries=true buffer-alignment=64' --canonicalize --cse --linalg-to-boscame='enable-spm-dma=true enable-ame=true store-policy=spm-dma dma-abi=args_aligned64_nonalias plan-report=true' 2>&1 | FileCheck %s --check-prefix=NO-DYNAMIC-DMA

module {
  tt.func public @dynamic_prefix(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %limit: i32) {
    %range = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32>
    %bound = tt.splat %limit : i32 -> tensor<32xi32>
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

// FACTS: event id=0 kind=load {{.*}}access-mask=arith.cmpi{{.*}}mask-cmp=slt mask-lhs=tt.make_range mask-rhs=tt.splat(arg#2)
// FACTS: event id=1 kind=store {{.*}}access-mask=arith.cmpi mask-cmp=slt mask-lhs=tt.make_range mask-rhs=tt.splat(arg#2)
// FACTS: memory-event {{.*}}kind = "load"{{.*}}origin_id = 0 : i64{{.*}}valid_shape = array<i64: -9223372036854775808>
// FACTS: memory-event {{.*}}kind = "store"{{.*}}origin_id = 1 : i64{{.*}}valid_shape = array<i64: -9223372036854775808>

// IR: %[[LIMIT:.*]] = arith.index_cast %arg2 : i32 to index
// IR: %[[CLAMPED:.*]] = arith.minsi %[[LIMIT]], %{{.*}} : index
// IR: %[[SIZE:.*]] = arith.maxsi %[[CLAMPED]], %{{.*}} : index
// IR: memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
// IR: memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
// IR: memref.copy {{.*}}bosc_ame.memory = {{.*}}kind = "load"{{.*}}valid_shape = array<i64: -9223372036854775808>
// IR: memref.subview %{{.*}}[0] [%[[SIZE]]] [1]
// IR: memref.copy {{.*}}bosc_ame.memory = {{.*}}kind = "store"{{.*}}valid_shape = array<i64: -9223372036854775808>

// NO-DYNAMIC-DMA: dynamic BOSCAME prefix extent must be clamped to the static maximum and provably divisible by a DMA packet
