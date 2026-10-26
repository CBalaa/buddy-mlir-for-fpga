// RUN: split-file %s %t
// RUN: buddy-opt %t/supported.mlir --linalg-to-boscame -o %t/supported.out
// RUN: FileCheck %s --check-prefix=SUPPORTED --implicit-check-not=linalg.reduce < %t/supported.out
// RUN: FileCheck %s --check-prefix=SCALAR < %t/supported.out
// RUN: not buddy-opt %t/unsupported.mlir --linalg-to-boscame 2>&1 | FileCheck %s --check-prefix=BAD-COMBINER
// RUN: not buddy-opt %t/complex.mlir --linalg-to-boscame 2>&1 | FileCheck %s --check-prefix=COMPLEX
// RUN: not buddy-opt %t/fast-maximum.mlir --linalg-to-boscame 2>&1 | FileCheck %s --check-prefix=FAST
// RUN: not buddy-opt %t/fast-maxnum.mlir --linalg-to-boscame 2>&1 | FileCheck %s --check-prefix=FAST

// RUN: buddy-opt %t/automatic.mlir --kernel-contract-annotate --bosc-ame-plan --bosc-ame-plan-verify -o %t/planned.mlir
// RUN: FileCheck %s --check-prefix=PLAN < %t/planned.mlir
// RUN: sed -e 's/ordered-scalar-max/ordered-vector-add/g' -e 's/vector_bytes = 4 : i64/vector_bytes = 64 : i64/g' %t/planned.mlir | not buddy-opt --bosc-ame-plan-verify 2>&1 | FileCheck %s --check-prefix=STRATEGY
// RUN: buddy-opt %t/automatic.mlir --linalg-to-boscame | FileCheck %s --check-prefix=AUTO --implicit-check-not=linalg.reduce
// RUN: buddy-opt %t/automatic.mlir --linalg-to-boscame --lower-bosc-ame -o /dev/null

// Read SPM through the vector port. maximumf retains the original scalar
// operation and argument order: it must not silently become maxnumf/vfredmax,
// whose NaN semantics differ. maxnumf keeps the vector reduction path for both
// commutative argument orders. Every reduction retains the caller-provided
// initial value. 32 elements also span more than one 64-byte packet.

// SCALAR-LABEL: func.func @maximum_input_first(
// SCALAR-NOT: vector.reduction
// SCALAR-LABEL: func.func @maximum_accumulator_first(
// SCALAR-NOT: vector.reduction
// SCALAR-LABEL: func.func @maxnum_input_first(

// SUPPORTED-LABEL: func.func @maximum_input_first(
// SUPPORTED: %[[OUT:.*]] = memref.alloca(){{.*}} : memref<f32>
// SUPPORTED: %[[INIT:.*]] = memref.load %[[OUT]][] : memref<f32>
// SUPPORTED: %[[RESULT:.*]] = scf.for %[[IV:.*]] = {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f32) {
// SUPPORTED: %[[VECTOR:.*]] = vector.load {{.*}}[%[[IV]]] : memref<32xf32, {{.*}}#bosc_ame.spm<0>>, vector<1xf32>
// SUPPORTED: %[[ELEMENT:.*]] = vector.extract %[[VECTOR]][0]
// SUPPORTED: %[[NEXT:.*]] = arith.maximumf %[[ELEMENT]], %[[ACC]] : f32
// SUPPORTED: scf.yield %[[NEXT]] : f32
// SUPPORTED: memref.store %[[RESULT]], %[[OUT]][]{{.*}} : memref<f32>
// SUPPORTED: return

// SUPPORTED-LABEL: func.func @maximum_accumulator_first(
// SUPPORTED: %[[OUT:.*]] = memref.alloca(){{.*}} : memref<f32>
// SUPPORTED: %[[INIT:.*]] = memref.load %[[OUT]][] : memref<f32>
// SUPPORTED: %[[RESULT:.*]] = scf.for %[[IV:.*]] = {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f32) {
// SUPPORTED: %[[VECTOR:.*]] = vector.load {{.*}}[%[[IV]]] : memref<32xf32, {{.*}}#bosc_ame.spm<0>>, vector<1xf32>
// SUPPORTED: %[[ELEMENT:.*]] = vector.extract %[[VECTOR]][0]
// SUPPORTED: %[[NEXT:.*]] = arith.maximumf %[[ACC]], %[[ELEMENT]] : f32
// SUPPORTED: scf.yield %[[NEXT]] : f32
// SUPPORTED: memref.store %[[RESULT]], %[[OUT]][]{{.*}} : memref<f32>
// SUPPORTED: return

// SUPPORTED-LABEL: func.func @maxnum_input_first(
// SUPPORTED: %[[OUT:.*]] = memref.alloca(){{.*}} : memref<f32>
// SUPPORTED: %[[INIT:.*]] = memref.load %[[OUT]][] : memref<f32>
// SUPPORTED: %[[RESULT:.*]] = scf.for %[[IV:.*]] = {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f32) {
// SUPPORTED: %[[VECTOR:.*]] = vector.load {{.*}}[%[[IV]]] : memref<32xf32, {{.*}}#bosc_ame.spm<0>>, vector<16xf32>
// SUPPORTED: %[[NEXT:.*]] = vector.reduction <maxnumf>, %[[VECTOR]], %[[ACC]] : vector<16xf32> into f32
// SUPPORTED: scf.yield %[[NEXT]] : f32
// SUPPORTED: memref.store %[[RESULT]], %[[OUT]][]{{.*}} : memref<f32>
// SUPPORTED: return

// SUPPORTED-LABEL: func.func @maxnum_accumulator_first(
// SUPPORTED: %[[OUT:.*]] = memref.alloca(){{.*}} : memref<f32>
// SUPPORTED: %[[INIT:.*]] = memref.load %[[OUT]][] : memref<f32>
// SUPPORTED: %[[RESULT:.*]] = scf.for %[[IV:.*]] = {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f32) {
// SUPPORTED: %[[VECTOR:.*]] = vector.load {{.*}}[%[[IV]]] : memref<32xf32, {{.*}}#bosc_ame.spm<0>>, vector<16xf32>
// SUPPORTED: %[[NEXT:.*]] = vector.reduction <maxnumf>, %[[VECTOR]], %[[ACC]] : vector<16xf32> into f32
// SUPPORTED: scf.yield %[[NEXT]] : f32
// SUPPORTED: memref.store %[[RESULT]], %[[OUT]][]{{.*}} : memref<f32>
// SUPPORTED: return

// BAD-COMBINER: error: v1 reduction plan requires ordered addf, maximumf, or maxnumf
// COMPLEX: error: v1 reduction plan requires ordered addf, maximumf, or maxnumf
// FAST: error: v1 reduction plan requires ordered addf, maximumf, or maxnumf
// STRATEGY: error: invalid BOSCAME plan: unsupported ordered reduction plan

//--- supported.mlir
module {
  func.func @maximum_input_first(%initial: f32) -> f32 {
    %output = memref.alloc() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%output : memref<f32>)
    %input = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<32xf32, #bosc_ame.spm<0>>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%input : memref<32xf32, #bosc_ame.spm<0>>)
    linalg.reduce ins(%input : memref<32xf32, #bosc_ame.spm<0>>)
      outs(%output : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
      %next = arith.maximumf %value, %acc : f32
      linalg.yield %next : f32
    }
    %result = memref.load %output[] : memref<f32>
    memref.dealloc %output : memref<f32>
    return %result : f32
  }

  func.func @maximum_accumulator_first(%initial: f32) -> f32 {
    %output = memref.alloc() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%output : memref<f32>)
    %input = bosc_ame.spm.view {bank = 0 : i32, offset = 128 : i64}
      : memref<32xf32, #bosc_ame.spm<0>>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%input : memref<32xf32, #bosc_ame.spm<0>>)
    linalg.reduce ins(%input : memref<32xf32, #bosc_ame.spm<0>>)
      outs(%output : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
      %next = arith.maximumf %acc, %value : f32
      linalg.yield %next : f32
    }
    %result = memref.load %output[] : memref<f32>
    memref.dealloc %output : memref<f32>
    return %result : f32
  }

  func.func @maxnum_input_first(%initial: f32) -> f32 {
    %output = memref.alloc() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%output : memref<f32>)
    %input = bosc_ame.spm.view {bank = 0 : i32, offset = 256 : i64}
      : memref<32xf32, #bosc_ame.spm<0>>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%input : memref<32xf32, #bosc_ame.spm<0>>)
    linalg.reduce ins(%input : memref<32xf32, #bosc_ame.spm<0>>)
      outs(%output : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
      %next = arith.maxnumf %value, %acc : f32
      linalg.yield %next : f32
    }
    %result = memref.load %output[] : memref<f32>
    memref.dealloc %output : memref<f32>
    return %result : f32
  }

  func.func @maxnum_accumulator_first(%initial: f32) -> f32 {
    %output = memref.alloc() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%output : memref<f32>)
    %input = bosc_ame.spm.view {bank = 0 : i32, offset = 384 : i64}
      : memref<32xf32, #bosc_ame.spm<0>>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%input : memref<32xf32, #bosc_ame.spm<0>>)
    linalg.reduce ins(%input : memref<32xf32, #bosc_ame.spm<0>>)
      outs(%output : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
      %next = arith.maxnumf %acc, %value : f32
      linalg.yield %next : f32
    }
    %result = memref.load %output[] : memref<f32>
    memref.dealloc %output : memref<f32>
    return %result : f32
  }
}

//--- unsupported.mlir
module {
  func.func @unsupported(%initial: f32) -> f32 {
    %output = memref.alloc() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%output : memref<f32>)
    %input = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<32xf32, #bosc_ame.spm<0>>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%input : memref<32xf32, #bosc_ame.spm<0>>)
    linalg.reduce ins(%input : memref<32xf32, #bosc_ame.spm<0>>)
      outs(%output : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
      %next = arith.mulf %value, %acc : f32
      linalg.yield %next : f32
    }
    %result = memref.load %output[] : memref<f32>
    memref.dealloc %output : memref<f32>
    return %result : f32
  }
}

//--- complex.mlir
module {
  func.func @complex(%initial: f32) -> f32 {
    %output = memref.alloc() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%output : memref<f32>)
    %input = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<32xf32, #bosc_ame.spm<0>>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%input : memref<32xf32, #bosc_ame.spm<0>>)
    linalg.reduce ins(%input : memref<32xf32, #bosc_ame.spm<0>>)
      outs(%output : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
      %squared = arith.mulf %value, %value : f32
      %next = arith.maximumf %squared, %acc : f32
      linalg.yield %next : f32
    }
    %result = memref.load %output[] : memref<f32>
    memref.dealloc %output : memref<f32>
    return %result : f32
  }
}

//--- fast-maximum.mlir
module {
  func.func @fast_maximum(%initial: f32) -> f32 {
    %output = memref.alloc() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%output : memref<f32>)
    %input = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<32xf32, #bosc_ame.spm<0>>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%input : memref<32xf32, #bosc_ame.spm<0>>)
    linalg.reduce ins(%input : memref<32xf32, #bosc_ame.spm<0>>)
      outs(%output : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
      %next = arith.maximumf %value, %acc fastmath<nnan> : f32
      linalg.yield %next : f32
    }
    %result = memref.load %output[] : memref<f32>
    memref.dealloc %output : memref<f32>
    return %result : f32
  }
}

//--- fast-maxnum.mlir
module {
  func.func @fast_maxnum(%initial: f32) -> f32 {
    %output = memref.alloc() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%output : memref<f32>)
    %input = bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
      : memref<32xf32, #bosc_ame.spm<0>>
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%input : memref<32xf32, #bosc_ame.spm<0>>)
    linalg.reduce ins(%input : memref<32xf32, #bosc_ame.spm<0>>)
      outs(%output : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
      %next = arith.maxnumf %value, %acc fastmath<nsz> : f32
      linalg.yield %next : f32
    }
    %result = memref.load %output[] : memref<f32>
    memref.dealloc %output : memref<f32>
    return %result : f32
  }
}

// Private input storage is selected by the planner and populated by DMA. This
// exercises the public pipeline in addition to fixed-window compatibility.
// PLAN-LABEL: func.func @automatic_maximum(
// PLAN: plan.buffer = {bank = {{[01]}} : i64, capacity = 128 : i64, level = "spm"
// PLAN: plan.reduce = {
// PLAN-SAME: strategy = "ordered-scalar-max"
// PLAN-SAME: vector_bytes = 4 : i64
// AUTO-LABEL: func.func @automatic_maximum(
// AUTO: bosc_ame.dma.start
// AUTO: bosc_ame.dma.wait
// AUTO: vector.load {{.*}} : memref<32xf32, {{.*}}#bosc_ame.spm<{{[01]}}>>, vector<1xf32>
// AUTO: vector.extract
// AUTO: arith.maximumf {{.*}} : f32
// AUTO: return
// PLAN-LABEL: func.func @automatic_maxnum(
// PLAN: plan.buffer = {bank = {{[01]}} : i64, capacity = 128 : i64, level = "spm"
// PLAN: plan.reduce = {
// PLAN-SAME: strategy = "ordered-vector-maxnum"
// PLAN-SAME: vector_bytes = 64 : i64
// AUTO-LABEL: func.func @automatic_maxnum(
// AUTO: bosc_ame.dma.start
// AUTO: bosc_ame.dma.wait
// AUTO: vector.load {{.*}} : memref<32xf32, {{.*}}#bosc_ame.spm<{{[01]}}>>, vector<16xf32>
// AUTO: vector.reduction <maxnumf>, {{.*}} : vector<16xf32> into f32
// AUTO: return

//--- automatic.mlir
module attributes {
  contract.abi = {args_aligned_bytes = 64 : i64, args_disjoint = true}
} {
  func.func @automatic_maximum(%source: memref<32xf32>, %initial: f32) -> f32 {
    %output = memref.alloc() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%output : memref<f32>)
    %input = memref.alloc() alignment = 64 : memref<32xf32>
    memref.copy %source, %input : memref<32xf32> to memref<32xf32>
    linalg.reduce ins(%input : memref<32xf32>)
      outs(%output : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
      %next = arith.maximumf %value, %acc : f32
      linalg.yield %next : f32
    }
    memref.dealloc %input : memref<32xf32>
    %result = memref.load %output[] : memref<f32>
    memref.dealloc %output : memref<f32>
    return %result : f32
  }

  func.func @automatic_maxnum(%source: memref<32xf32>, %initial: f32) -> f32 {
    %output = memref.alloc() : memref<f32>
    linalg.fill ins(%initial : f32) outs(%output : memref<f32>)
    %input = memref.alloc() alignment = 64 : memref<32xf32>
    memref.copy %source, %input : memref<32xf32> to memref<32xf32>
    linalg.reduce ins(%input : memref<32xf32>)
      outs(%output : memref<f32>) dimensions = [0]
      (%value: f32, %acc: f32) {
      %next = arith.maxnumf %value, %acc : f32
      linalg.yield %next : f32
    }
    memref.dealloc %input : memref<32xf32>
    %result = memref.load %output[] : memref<f32>
    memref.dealloc %output : memref<f32>
    return %result : f32
  }
}
