#ifndef STRUCTURED_TO_MEMREF_CONVERSION_PASSES
#define STRUCTURED_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMemref : Pass<"structured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
  let options = [
      Option<"enableTensorFirstVectorCpu", "enable-tensor-first-vector-cpu", "bool", /*default*/"false",
             "Enable tensor-first fast path for simple 1D contiguous load/store on CPU">
  ];
}

#endif
