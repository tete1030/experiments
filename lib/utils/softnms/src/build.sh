#!/bin/sh

python setup.py build_ext --inplace
mv softnms*.so ../softnms.so
rm -r softnms.c build
