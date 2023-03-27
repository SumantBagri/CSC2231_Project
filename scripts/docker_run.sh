#!/usr/bin/env bash

sudo docker run --runtime nvidia -it -p 8888:8888 -v ~/project:/opt/project --name ldm -d trapdoor20/csc2231:$1
sudo docker logs ldm --since 3s
