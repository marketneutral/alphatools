#!/usr/bin/env bash

source ~/.basrc
# must append to .bashrc
if [ "$MKL_THREADING_LAYER" = "" ]
then
    export MKL_THREADING_LAYER=GNU
    echo 'export MKL_THREADING_LAYER=GNU' >> ~/.bashrc
fi

if [ "$THEANO_FLAGS" = "" ]
then
    # Needed for Mac OS X
    export "THEANO_FLAGS='gcc.cxxflags=-Wno-c++11-narrowing'"
    echo "export \"THEANO_FLAGS='gcc.cxxflags=-Wno-c++11-narrowing'\"" >> ~/.bashrc
fi

