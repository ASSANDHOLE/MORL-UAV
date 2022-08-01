#!/bin/bash

set -e

if [ $# -eq 0 ]
then
  if [ ! -d "build" ]
  then
    mkdir build
  elif [ "$(ls -A ./build)" ]
  then
    echo "./build not empty. Continue? [y|n]"
    read -r empty_dir
    if ((empty_dir == "y" || empty_dir == "yes"))
    then
      rm -rf ./build
      mkdir build
    else
      exit 0
    fi
  fi
  path_to_dest=../gym_uav/env/
else
  path_to_dest=$1
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=release
make -j 8
cd ..
cp ./build/libc_functions.so "$path_to_dest"
echo "copy 'libc_functions.so' to $path_to_dest"