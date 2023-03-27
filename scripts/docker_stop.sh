
#!/usr/bin/env bash

sudo docker stop ldm
sudo docker rm $(sudo docker ps -a -q -f status=exited)
echo Stopped and cleaned all containers!
