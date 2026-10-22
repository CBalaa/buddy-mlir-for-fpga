// RUN: not buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga bm=0' 2>&1 | FileCheck %s --check-prefix=TILE
// RUN: not buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga bn=-1' 2>&1 | FileCheck %s --check-prefix=TILE
// RUN: not buddy-opt %s --lower-linalg-to-boscame='target=nr-fpga bk=0' 2>&1 | FileCheck %s --check-prefix=TILE
// RUN: not buddy-opt %s --lower-linalg-to-boscame='target=unknown' 2>&1 | FileCheck %s --check-prefix=TARGET

// Configuration errors are rejected even when there are no consumers.
// TILE: BOSCAME tile sizes must be positive
// TARGET: unsupported BOSCAME target; expected nr-fpga
module {
}
