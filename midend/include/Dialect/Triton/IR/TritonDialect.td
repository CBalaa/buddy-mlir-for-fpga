#ifndef TRITON_DIALECT
#define TRITON_DIALECT

include "mlir/IR/OpBase.td"

def Triton_Dialect : Dialect {
  let name = "tt";

  let cppNamespace = "::mlir::triton";

  let summary = "The Triton IR in MLIR";

  let description = [{
    Triton Dialect.

    Dependent Dialects:
      * Arith:
        * addf, addi, andi, cmpf, cmpi, divf, fptosi, ...
      * Math:
        * exp, sin, cos, log, ...
      * StructuredControlFlow:
        * for, if, while, yield, condition
      * ControlFlow:
        * br, cond_br
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "math::MathDialect",
    "scf::SCFDialect",
    "cf::ControlFlowDialect",
    "ub::UBDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();

    static TritonDialect *getLoaded(MLIRContext *ctx) {
      return ctx->getLoadedDialect<TritonDialect>();
    }
    static TritonDialect *getLoaded(Operation *op) {
      return getLoaded(op->getContext());
    }
  }];

  let discardableAttrs = (ins
     "::mlir::IntegerAttr":$num_stages,
     "::mlir::IntegerAttr":$latency,
     "::mlir::IntegerAttr":$self_latency
  );

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;

  // MLIR 24 always stores op attributes as properties, and by default it also
  // requires every declarative assembly format to bind each inherent attribute
  // (or to cover them with `prop-dict`).  TTIR's formats are inherited verbatim
  // from upstream Triton, which predates that rule: `tt.load`, `tt.dot`,
  // `tt.call`, ... all print their inherent attributes as part of `attr-dict`
  // (e.g. `tt.dot ... {inputPrecision = 2 : i32}`), and Triton-generated TTIR
  // files use that spelling.
  //
  // Opting into MLIR's legacy assembly-format behavior keeps that textual form
  // parseable and printable exactly as before.  Attributes are still stored as
  // properties internally, so this only affects the assembly format.
  let useStrictPropertiesInAssemblyFormat = 0;
}

include "Triton/IR/TritonTypes.td"


#endif // TRITON_DIALECT
