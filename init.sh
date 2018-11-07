#!/bin/bash

pip install -r requirements.txt
pushd pose/models/displace/src
./build.sh
popd
pushd utils/softnms/src
./build.sh
popd
