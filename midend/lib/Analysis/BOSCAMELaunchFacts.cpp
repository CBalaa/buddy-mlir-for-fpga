//===- BOSCAMELaunchFacts.cpp - BOSCAME launch metadata facts
//--------------===//

#include "Analysis/BOSCAMELaunchFacts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <utility>

using namespace mlir;

namespace buddy::boscame {

FailureOr<llvm::SmallVector<LaunchFunctionFact, 4>>
analyzeBOSCAMELaunchFacts(ModuleOp module) {
  llvm::SmallVector<LaunchFunctionFact, 4> functions;
  LogicalResult status = success();
  module.walk([&](func::FuncOp function) {
    LaunchFunctionFact fact;
    fact.functionName = function.getSymName().str();
    llvm::DenseSet<int64_t> seenAxes;
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      auto role =
          function.getArgAttrOfType<StringAttr>(index, "bosc_ame.program_role");
      if (!role || (role.getValue() != "num_programs" &&
                    role.getValue() != "program_id"))
        continue;

      LaunchProgramArgumentFact argument;
      argument.role = role.getValue() == "num_programs"
                          ? LaunchProgramRole::NumPrograms
                          : LaunchProgramRole::ProgramId;
      argument.argumentIndex = index;
      if (auto axis = function.getArgAttrOfType<IntegerAttr>(
              index, "bosc_ame.program_axis")) {
        argument.axis = axis.getInt();
        if (argument.role == LaunchProgramRole::NumPrograms &&
            (*argument.axis < 0 || *argument.axis >= 3 ||
             !seenAxes.insert(*argument.axis).second)) {
          function.emitError("invalid BOSCAME num_programs axis; expected "
                             "unique axes in [0,2]");
          status = failure();
        }
      }
      fact.programArguments.push_back(std::move(argument));
    }

    if (Attribute attr = function->getAttr("gpu.known_grid_size")) {
      auto grid = dyn_cast<DenseI32ArrayAttr>(attr);
      if (!grid || grid.size() != 3 ||
          llvm::any_of(grid.asArrayRef(),
                       [](int32_t extent) { return extent <= 0; })) {
        function.emitError("invalid gpu.known_grid_size contract; expected "
                           "three positive i32 extents");
        status = failure();
      } else {
        fact.knownGrid = std::array<int32_t, 3>{grid[0], grid[1], grid[2]};
      }
    }
    functions.push_back(std::move(fact));
  });
  if (failed(status))
    return failure();
  return functions;
}

} // namespace buddy::boscame
