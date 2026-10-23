//===- BOSCAMEBoundaryTransfers.h - Resident external DMA plans -*- C++ -*-===//

#ifndef BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOUNDARY_TRANSFERS_H
#define BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOUNDARY_TRANSFERS_H

#include "Conversion/LowerLinalgToBOSCAME/BOSCAMEFootprint.h"

namespace buddy::boscame {

/// Prove an identity copy between an external DDR descriptor and an existing
/// resident SPM window. Packet offsets are relative to the two descriptors;
/// the root window's bank/offset/capacity remain the placement reservation.
/// This query neither mutates IR nor diagnoses an unsupported candidate, so
/// placement preflight can use exactly the transfer planner's legality proof.
std::optional<TransferPlan>
planResidentBoundaryTransfer(mlir::Operation *operation,
                             const TargetInfo &target);

} // namespace buddy::boscame

#endif // BUDDY_CONVERSION_LOWERLINALGTOBOSCAME_BOUNDARY_TRANSFERS_H
