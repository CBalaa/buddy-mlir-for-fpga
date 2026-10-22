//===- BOSCAMELaunchFacts.h - BOSCAME launch metadata facts -------*- C++
//-*-===//

#ifndef BUDDY_ANALYSIS_BOSCAMELAUNCHFACTS_H
#define BUDDY_ANALYSIS_BOSCAMELAUNCHFACTS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace buddy::boscame {

enum class LaunchProgramRole { NumPrograms, ProgramId };

/// A runtime program parameter identified by explicit frontend metadata.
/// A missing axis remains unknown; argument position alone never assigns one.
struct LaunchProgramArgumentFact {
  LaunchProgramRole role = LaunchProgramRole::NumPrograms;
  unsigned argumentIndex = 0;
  std::optional<int64_t> axis;
};

/// Detached launch metadata with no operation, value, type, or attribute
/// handles. knownGrid is a caller-provided precondition, not a proof that the
/// runtime arguments satisfy it or authorization for cross-program reuse.
struct LaunchFunctionFact {
  std::string functionName;
  std::optional<std::array<int32_t, 3>> knownGrid;
  llvm::SmallVector<LaunchProgramArgumentFact, 6> programArguments;
};

/// Collect only explicit function/argument launch metadata without changing IR.
/// Reject malformed known-grid contracts and duplicate/out-of-range annotated
/// num_programs axes; preserve the existing unknown-axis and unknown-role
/// cases.
mlir::FailureOr<llvm::SmallVector<LaunchFunctionFact, 4>>
analyzeBOSCAMELaunchFacts(mlir::ModuleOp module);

} // namespace buddy::boscame

#endif // BUDDY_ANALYSIS_BOSCAMELAUNCHFACTS_H
