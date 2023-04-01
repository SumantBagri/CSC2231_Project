#!/usr/bin/env bash

docker run --runtime nvidia -it --network host -v ~/project:/opt/project --name ldm -d trapdoor20/csc2231:$1
docker logs ldm --since 3s
