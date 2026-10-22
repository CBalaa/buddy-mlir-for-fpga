// RUN: split-file %s %t
// RUN: not buddy-opt %t/tensor.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s
// RUN: not buddy-opt %t/unused-tensor.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s
// RUN: not buddy-opt %t/external-tensor.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s
// RUN: not buddy-opt %t/bridge.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=BRIDGE
// RUN: not buddy-opt %t/unrealized.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s
// RUN: not buddy-opt %t/tensor.mlir --linalg-to-boscame 2>&1 | FileCheck %s
// RUN: not buddy-opt %t/unrealized.mlir --linalg-to-boscame 2>&1 | FileCheck %s

// The backend consumes an explicitly bufferized program.  It must reject
// incomplete bufferization without inserting a second implicit bufferization
// pipeline or accepting a cast as a data movement operation.
// BRIDGE: incomplete BOSCAME lowering: residual Triton or bufferization bridge operation
// CHECK: input must be pure buffer-form Linalg
// CHECK-SAME: without tensor SSA or unrealized_conversion_cast

//--- tensor.mlir
module {
  func.func @tensor(%a: tensor<16xf32>) -> tensor<16xf32> {
    return %a : tensor<16xf32>
  }
}

//--- bridge.mlir
module {
  func.func @bridge(%a: memref<16xf32>) -> tensor<16xf32> {
    %tensor = bufferization.to_tensor %a restrict : memref<16xf32> to tensor<16xf32>
    return %tensor : tensor<16xf32>
  }
}

//--- unused-tensor.mlir
module {
  func.func @unused_tensor(%unused: tensor<16xf32>) {
    return
  }
}

//--- unrealized.mlir
module {
  func.func @unrealized(%a: memref<16xf32>) {
    %cast = builtin.unrealized_conversion_cast %a : memref<16xf32> to memref<16xf32>
    return
  }
}

//--- external-tensor.mlir
module {
  func.func private @external_tensor(tensor<16xf32>) -> tensor<16xf32>
}
