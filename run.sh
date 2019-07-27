#!/usr/bin/env bash

set -o xtrace

HOST=$1
EXPERIMENT=$2

rsync -avP ./ ${HOST}:openmc --exclude cmake-build-debug --exclude build \
      --exclude .git --exclude endf71_hdf5 --exclude '*.ppm' --exclude openmc/capi/libopenmc.dylib \
      --exclude *.blend* --exclude *.trelis* --exclude *.stl --exclude geometry-no-boundary.obj \
      --exclude '*.h5' --exclude '*.vti' --exclude '*.vtk' --exclude '*.stl'

ssh ${HOST} -t "make -j 16 --directory=openmc/build && cd openmc/experiments/${EXPERIMENT} && ~/.local/bin/openmc ${@:3}"
