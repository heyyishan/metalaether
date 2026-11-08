#!/usr/bin/env bash
set -euo pipefail
if python3 -m pybind11 --cmakedir >/dev/null 2>&1; then
  PYBIND11_DIR="$(python3 -m pybind11 --cmakedir)"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$PYBIND11_DIR"
else
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
fi
cmake --build build --parallel
