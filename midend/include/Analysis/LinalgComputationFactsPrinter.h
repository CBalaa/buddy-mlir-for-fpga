//===- LinalgComputationFactsPrinter.h - SSA fact diagnostics -----*- C++
//-*-===//

#ifndef BUDDY_ANALYSIS_LINALGCOMPUTATIONFACTSPRINTER_H
#define BUDDY_ANALYSIS_LINALGCOMPUTATIONFACTSPRINTER_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace buddy::boscame {
struct LinalgComputationFacts;

/// Print the explicit SSA graph without inferring execution order, buffer
/// contents, reduction identities, or placement legality from its edges.
void printLinalgComputationFacts(const LinalgComputationFacts &facts,
                                 llvm::raw_ostream &out,
                                 llvm::StringRef indent = "");
} // namespace buddy::boscame

#endif // BUDDY_ANALYSIS_LINALGCOMPUTATIONFACTSPRINTER_H
