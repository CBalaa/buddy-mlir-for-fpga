// RUN: split-file %s %t
// RUN: buddy-opt %t/success.mlir --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t/success.out 2>%t/success.plan
// RUN: FileCheck %s --check-prefix=SUCCESS < %t/success.plan
// RUN: buddy-opt %t/group-capacity.mlir --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t/capacity.out 2>%t/capacity.plan
// RUN: FileCheck %s --check-prefix=CAPACITY < %t/capacity.plan
// RUN: buddy-opt %t/staging-capacity.mlir --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t/staging.out 2>%t/staging.plan
// RUN: FileCheck %s --check-prefix=STAGING < %t/staging.plan
// RUN: buddy-opt %t/computation-legality.mlir --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t/legality.out 2>%t/legality.plan
// RUN: FileCheck %s --check-prefix=LEGALITY < %t/legality.plan

// SUCCESS: buffer-placement: resident=2 ddr=0 applied=true
// SUCCESS-COUNT-2: kind=SpmResident bytes=64
// SUCCESS: resources: transfers=0 dma-packets=0 read-bytes=0 write-bytes=0

// CAPACITY: buffer-placement: resident=0 ddr=5 applied=false
// CAPACITY-COUNT-5: kind=StagedFallback bytes=65536{{.*}}group=0 reason=spm-capacity applied=false

// STAGING: buffer-placement: resident=0 ddr=2 applied=false
// STAGING-COUNT-2: kind=StagedFallback bytes=65536{{.*}}group=0 reason=missing-staging-capacity applied=false
// STAGING-SAME: detail=resident + explicit windows leave insufficient per-op staging capacity

// LEGALITY: buffer-placement: resident=0 ddr=2 applied=false
// LEGALITY-COUNT-2: kind=StagedFallback{{.*}}group=0 reason=computation-legality applied=false

//--- success.mlir
module {
  func.func @resident_pair() {
    %zero = arith.constant 0.0 : f32
    %a = memref.alloc() alignment = 64 : memref<16xf32>
    %b = memref.alloc() alignment = 64 : memref<16xf32>
    linalg.fill ins(%zero : f32) outs(%a : memref<16xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a : memref<16xf32>) outs(%b : memref<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
}

//--- computation-legality.mlir
// Each root is a valid static vector buffer, but their connected operation
// changes element width. The SPM vector island cannot lower this operation;
// detached preflight must reject both roots before either allocation changes.
module {
  func.func @whole_group_vector_legality() {
    %zero = arith.constant 0.0 : f32
    %a = memref.alloc() alignment = 64 : memref<16xf32>
    %b = memref.alloc() alignment = 64 : memref<16xf64>
    linalg.fill ins(%zero : f32) outs(%a : memref<16xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a : memref<16xf32>) outs(%b : memref<16xf64>) {
    ^bb0(%in: f32, %out: f64):
      %wide = arith.extf %in : f32 to f64
      linalg.yield %wide : f64
    }
    return
  }
}

//--- group-capacity.mlir
// Five fixed windows exceed the two banks. Each individual pair still fits
// the original per-op staging, so the connected group must fall back intact.
module {
  func.func @resident_group_exceeds_capacity() {
    %zero = arith.constant 0.0 : f32
    %a = memref.alloc() alignment = 64 : memref<16384xf32>
    %b = memref.alloc() alignment = 64 : memref<16384xf32>
    %c = memref.alloc() alignment = 64 : memref<16384xf32>
    %d = memref.alloc() alignment = 64 : memref<16384xf32>
    %e = memref.alloc() alignment = 64 : memref<16384xf32>
    linalg.fill ins(%zero : f32) outs(%a : memref<16384xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a : memref<16384xf32>) outs(%b : memref<16384xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%b : memref<16384xf32>) outs(%c : memref<16384xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%c : memref<16384xf32>) outs(%d : memref<16384xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%d : memref<16384xf32>) outs(%e : memref<16384xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    return
  }
}

//--- staging-capacity.mlir
// The two resident roots need only 128 KiB, but all three external operands
// of the last op also need 192 KiB staging. The original last op fits exactly
// in two banks, so failed prospective placement must retain that valid path.
module {
  func.func @staging_in_same_budget(%x: memref<16384xf32>,
      %y: memref<16384xf32>, %z: memref<16384xf32>) {
    %zero = arith.constant 0.0 : f32
    %a = memref.alloc() alignment = 64 : memref<16384xf32>
    %b = memref.alloc() alignment = 64 : memref<16384xf32>
    linalg.fill ins(%zero : f32) outs(%a : memref<16384xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%a : memref<16384xf32>) outs(%b : memref<16384xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
        affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%x, %y, %z : memref<16384xf32>, memref<16384xf32>, memref<16384xf32>)
      outs(%b : memref<16384xf32>) {
    ^bb0(%xv: f32, %yv: f32, %zv: f32, %out: f32):
      %sum = arith.addf %xv, %yv : f32
      %result = arith.addf %sum, %zv : f32
      linalg.yield %result : f32
    }
    return
  }
}
