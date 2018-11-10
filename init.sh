#!/bin/bash

pushd pose/models/displace/src
./build.sh
popd
pushd utils/softnms/src
./build.sh
popd
