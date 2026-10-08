module {
func.func @kernel_dequantize_3x19(%x:memref<3x19xi32>, %row:memref<3xf32>, %col:memref<19xf32>, %y:memref<3x19xf32>) attributes {llvm.emit_c_interface} {
 linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i)>,affine_map<(i,j)->(j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%x,%row,%col:memref<3x19xi32>,memref<3xf32>,memref<19xf32>) outs(%y:memref<3x19xf32>) {
 ^bb0(%xv:i32,%rs:f32,%cs:f32,%unused:f32):
  %f = arith.sitofp %xv : i32 to f32
  %a = arith.mulf %f,%rs : f32
  %b = arith.mulf %a,%cs : f32
  linalg.yield %b : f32
 }
 return
}
}
