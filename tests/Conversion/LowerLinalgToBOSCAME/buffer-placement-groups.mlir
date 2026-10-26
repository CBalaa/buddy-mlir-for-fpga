// RUN: split-file %s %t
// RUN: buddy-opt %t/success.mlir --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t/success.out 2>%t/success.plan
// RUN: FileCheck %s --check-prefix=SUCCESS < %t/success.plan
// RUN: not buddy-opt %t/group-capacity.mlir --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t/capacity.out 2>%t/capacity.plan
// RUN: FileCheck %s --check-prefix=CAPACITY < %t/capacity.plan
// RUN: buddy-opt %t/staging-capacity.mlir --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t/staging.out 2>%t/staging.plan
// RUN: FileCheck %s --check-prefix=STAGING < %t/staging.plan
// RUN: not buddy-opt %t/computation-legality.mlir --linalg-to-boscame='buffer-placement=true enable-spm-dma=true dma-abi=args_aligned64_nonalias plan-report=true' -o %t/legality.out 2>%t/legality.plan
// RUN: FileCheck %s --check-prefix=LEGALITY < %t/legality.plan

// SUCCESS-COUNT-2: memref.alloc plan.buffer=
// SUCCESS: high_water = array<i64: 64, 64>
// SUCCESS-NOT: plan.transfer

// Capacity uses actual reserved windows. V1 does not invent private DDR
// staging when the five roots exceed both banks.
// CAPACITY: SPM workspace capacity exceeded after legal direct/staging alternatives

// Existing DDR vector operands use direct RVV reads and need no staging
// windows. Two 64 KiB roots plus the preserved arithmetic intermediate fit.
// STAGING: high_water = array<i64: 131072, 65536>
// STAGING-NOT: plan.transfer

// LEGALITY: unsupported lane-wise scalar expression

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
      %negated = arith.negf %in : f32
      linalg.yield %negated : f32
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
// Five private windows exceed both banks. No v1 private DDR fallback exists.
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
      %negated = arith.negf %in : f32
      linalg.yield %negated : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%b : memref<16384xf32>) outs(%c : memref<16384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %negated = arith.negf %in : f32
      linalg.yield %negated : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%c : memref<16384xf32>) outs(%d : memref<16384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %negated = arith.negf %in : f32
      linalg.yield %negated : f32
    }
    linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]}
      ins(%d : memref<16384xf32>) outs(%e : memref<16384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %negated = arith.negf %in : f32
      linalg.yield %negated : f32
    }
    return
  }
}

//--- staging-capacity.mlir
// External RVV inputs remain in DDR. The original arithmetic order is kept
// through an additional private intermediate, with no external staging.
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
      %negated = arith.negf %in : f32
      linalg.yield %negated : f32
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
