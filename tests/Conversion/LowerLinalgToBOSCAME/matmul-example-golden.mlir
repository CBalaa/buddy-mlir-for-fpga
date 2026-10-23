// RUN: split-file %S/Inputs/matmul-example-golden.mlir %t
// RUN: buddy-opt %t/kernel.bufferized.mlir --linalg-to-boscame -o %t/default.mlir
// RUN: cmp %t/kernel.boscame.mlir %t/default.mlir
// RUN: buddy-opt %t/kernel.bufferized.mlir '--linalg-to-boscame=buffer-placement=true' -o %t/resident.mlir
// RUN: cmp %t/kernel.boscame.mlir %t/resident.mlir

// Exact pass output from the clean, reproducible matmul_i8_128x1024x128
// example build. The AME direct routes and accumulator fusion leave no
// compiler-managed roots for buffer placement to change. Keep the raw input
// and golden in Inputs so the fixture needs no Triton or ELF toolchain.
