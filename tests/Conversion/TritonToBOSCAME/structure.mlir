// The pass has to produce the three-level mapping the task asks for: DDR ->
// SPM through real DMA transfers, SPM -> AME tile registers, and the AME
// result back through SPM to DDR.  These checks pin the shape of that IR; the
// numeric and memory behaviour is verified by the `numeric-*.py` tests.

// RUN: buddy-opt %S/Inputs/single-dot-64x64x64.ttir \
// RUN:   --triton-to-boscame=dma-abi=args_aligned64_nonalias \
// RUN:   | FileCheck %s --check-prefix=DENSE

// The DMA contract lives on the module.
// DENSE: module attributes {bosc_ame.dma.abi = "args_aligned64_nonalias", bosc_ame.target = "nr-fpga"}

// The kernel ABI grows by the program information arguments exactly once.
// DENSE: func.func @single_dot64(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %arg3: memref<?xi8>, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32)
// DENSE-SAME: bosc_ame.scratch_bytes = 16384 : i64
// DENSE-SAME: llvm.emit_c_interface

// The SPM workspace is the fixed pair of bank windows.
// DENSE: bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
// DENSE: bosc_ame.spm.view {bank = 1 : i32, offset = 0 : i64}

// Staging goes through DMA with a token and a matching wait.
// DENSE: bosc_ame.dma.start
// DENSE-NEXT: bosc_ame.dma.wait

// The accumulator is seeded in SPM, A/B enter the tile registers, the single
// 64-wide block is one mma, and the result goes back through SPM.
// DENSE: bosc_ame.tile.load_acc
// DENSE: bosc_ame.tile.load_a
// DENSE: bosc_ame.tile.load_b
// DENSE: bosc_ame.tile.mma %{{.*}}, %{{.*}}, %{{.*}} : vector<64x64xi32>, vector<64x64xi8>, vector<64x64xi8> -> vector<64x64xi32>
// DENSE: bosc_ame.tile.store_acc

// No Triton operation is left behind.
// DENSE-NOT: tt.

// A reduction wider than the 64-wide AME tile must become a real 64-step
// `scf.for` that carries the accumulator through SSA, with the tile loads from
// SPM and one mma in the body.  No DMA may appear inside that loop: the whole
// accumulation is staged before the accumulator is loaded, because an AME
// register value cannot survive the DMA runtime call.
// RUN: buddy-opt %S/Inputs/k-blocked-64x64x256.ttir \
// RUN:   --triton-to-boscame=dma-abi=args_aligned64_nonalias -o - \
// RUN:   | FileCheck %s --check-prefix=KBLOCK
// KBLOCK: bosc_ame.spm.view {bank = 0 : i32, offset = 0 : i64}
// KBLOCK: bosc_ame.tile.load_acc
// KBLOCK: scf.for {{.*}} step %c64 iter_args({{.*}}) -> (vector<64x64xi32>) {
// KBLOCK-NOT: bosc_ame.dma.start
// KBLOCK: bosc_ame.tile.load_a
// KBLOCK: bosc_ame.tile.load_b
// KBLOCK: bosc_ame.tile.mma {{.*}} : vector<64x64xi32>, vector<64x64xi8>, vector<64x64xi8> -> vector<64x64xi32>
// KBLOCK: scf.yield
// KBLOCK: bosc_ame.tile.store_acc

// A 63-byte original row cannot be transferred directly: pack valid elements
// into a zero-filled DDR tile, then DMA complete rows into SPM.
// RUN: buddy-opt %S/Inputs/masked-a-stride128.ttir \
// RUN:   --triton-to-boscame=dma-abi=args_aligned64_nonalias -o - \
// RUN:   | FileCheck %s --check-prefix=MASKED
// MASKED: linalg.fill ins(%{{.*}} : i8) outs(%{{.*}} : memref<64x64xi8
// MASKED: scf.for
// MASKED: memref.load %{{.*}}[%{{.*}}] : memref<64xi8>
// MASKED: memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<64x64xi8>
// MASKED: bosc_ame.dma.start
// MASKED-NEXT: bosc_ame.dma.wait
// MASKED: bosc_ame.tile.load_a
