#!/bin/bash

python setup.py build --build-lib ./build
shopt -s extglob
cp "./build/displace_cuda."@(*)".so" ../displace_cuda.so
rm -r build
