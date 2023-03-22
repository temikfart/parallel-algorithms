#!/usr/bin/env sh

PROGRAM="../cmake-build-debug/hw-1/hw-1"
for i in $(seq 1 8); do
  echo "#Proc = $i"
  chmod 0755 $PROGRAM
  mpiexec -np $i $PROGRAM
  echo "======================================"
done