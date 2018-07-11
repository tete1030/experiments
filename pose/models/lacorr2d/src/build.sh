#!/bin/bash

python setup.py build --build-lib ./build
shopt -s extglob
cp "./build/lacorr2d_cuda."@(*)".so" ../lacorr2d_cuda.so
rm -r build
