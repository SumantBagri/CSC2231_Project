
#!/usr/bin/env bash

docker stop ldm
docker rm $(docker ps -a -q -f status=exited)
echo Stopped and cleaned all containers!
