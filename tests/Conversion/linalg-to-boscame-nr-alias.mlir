// RUN: buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga' | FileCheck %s

// C is a view of the same byte buffer as A.  The NR lowering keeps C in the
// AME accumulator until the end of the K reduction, so this overlap must stay
// on the conservative linalg/CPU path.  In particular, the dynamic legality
// predicate must agree with the rewrite matcher and leave the operation legal.
func.func @destination_alias(%base: memref<256xi8>,
                             %b: memref<16x4xi8>) {
  %c0 = arith.constant 0 : index
  %a = memref.view %base[%c0][] : memref<256xi8> to memref<4x16xi8>
  %c = memref.view %base[%c0][] : memref<256xi8> to memref<4x4xi32>
  linalg.matmul ins(%a, %b : memref<4x16xi8>, memref<16x4xi8>)
      outs(%c : memref<4x4xi32>)
  return
}

// CHECK-LABEL: func.func @destination_alias
// CHECK: linalg.matmul
// CHECK-NOT: bosc_ame.
// CHECK: return
