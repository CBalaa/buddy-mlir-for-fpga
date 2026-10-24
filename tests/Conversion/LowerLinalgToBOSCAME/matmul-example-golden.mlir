// RUN: %python %S/Inputs/kernel-legacy-cases.py %s buddy-opt
// Original programs remain below. The harness checks explicit v1 limits and
// supported routes; see PlanRoutes.py, PlanEmitValidation.py,
// PlanTerminalStore.py, physical-adapter.py and tests/KernelContract/.

// Exact pass output from the clean, reproducible matmul_i8_128x1024x128
// example build. The AME direct routes and accumulator fusion leave no
// compiler-managed roots for buffer placement to change. Keep the raw input
// and golden in Inputs so the fixture needs no Triton or ELF toolchain.
