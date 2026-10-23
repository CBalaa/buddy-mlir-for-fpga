// RUN: buddy-opt %S/Inputs/layernorm-197x768-bufferized.mlir --linalg-to-boscame='buffer-placement=true plan-report=true' -o %t 2>%t.plan
// RUN: FileCheck %s --check-prefix=PLAN < %t.plan
// RUN: FileCheck %s --check-prefix=IR < %t
// RUN: FileCheck %s --check-prefix=ROOTS < %t
// RUN: buddy-opt %t --bosc-ame-spm-dma-check -o /dev/null

// This is the real bufferized frontend output. Its six 4 KiB floating roots
// include padding storage, broadcasts and squares. Broadcast forwarding removes
// their shared mean/invstd root after placement; the in-place data updates remain.
// PLAN: buffer-placement: resident=6 ddr=0 applied=true
// PLAN-COUNT-6: kind=SpmResident bytes=4096
// PLAN: resources: transfers=4 dma-packets=4 read-bytes=9216 write-bytes=3072
// IR-COUNT-4: bosc_ame.dma.start
// IR-NOT: bosc_ame.dma.start
// ROOTS-COUNT-5: bosc_ame.spm.view
// ROOTS-NOT: bosc_ame.spm.view
