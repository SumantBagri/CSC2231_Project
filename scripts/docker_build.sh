#!/usr/bin/env bash

docker build -f dockerfiles/Dockerfile.$1 -t trapdoor20/csc2231:$2 .
