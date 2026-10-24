// RUN: split-file %s %t
// RUN: not buddy-opt %t/signed.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=false dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=DISABLED
// RUN: not buddy-opt %t/unsigned.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=TYPE
// RUN: not buddy-opt %t/i16.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=TYPE
// RUN: not buddy-opt %t/floating.mlir --lower-linalg-to-boscame='enable-spm-dma=true enable-ame=true dma-abi=args_aligned64_nonalias target=nr-fpga' 2>&1 | FileCheck %s --check-prefix=TYPE

// A marker is an explicit AME obligation.  Unsupported integer interpretation
// and element types must fail instead of falling back to another arithmetic
// operation while leaving an apparently completed marker behind.
// DISABLED: remove disabled legacy feature overrides
// TYPE: legacy AME marker requires signed i8 x i8 -> i32 matmul

//--- signed.mlir
module {
  func.func @signed(%a: memref<64x64xi8>, %b: memref<64x64xi8>,
                    %c: memref<64x64xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>, memref<64x64xi8>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

//--- unsigned.mlir
module {
  func.func @unsigned(%a: memref<64x64xi8>, %b: memref<64x64xi8>,
                      %c: memref<64x64xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_unsigned>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi8>, memref<64x64xi8>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

//--- i16.mlir
module {
  func.func @i16(%a: memref<64x64xi16>, %b: memref<64x64xi16>,
                 %c: memref<64x64xi32>) {
    linalg.matmul {cast = #linalg.type_fn<cast_signed>, bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xi16>, memref<64x64xi16>)
      outs(%c : memref<64x64xi32>)
    return
  }
}

//--- floating.mlir
module {
  func.func @floating(%a: memref<64x64xf32>, %b: memref<64x64xf32>,
                      %c: memref<64x64xf32>) {
    linalg.matmul {bosc_ame.lower_to_ame}
      ins(%a, %b : memref<64x64xf32>, memref<64x64xf32>)
      outs(%c : memref<64x64xf32>)
    return
  }
}
