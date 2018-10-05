#!/usr/bin/env bash

# must append to .bashrc
source ~/.bashrc

if ( ! $?MKL_THREADING_LAYER ) then
   echo 'export MKL_THREADING_LAYER=GNU' >> ~/.bashrc;
endif

if ( ! $?THEANO_FLAGS ) then
   # Needed for Mac OS X
   echo "export THEANO_FLAGS='gcc.cxxflags=-Wno-c++11-narrowing'" >> ~./bashrc;
endif
   
source ~/.bashrc
