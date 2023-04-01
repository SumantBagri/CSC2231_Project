#!/usr/bin/env bash

set -e

source scripts/opencv_version.sh

#
# PyTorch 
#
build_pytorch()
{
	local pytorch_url=$1
	local pytorch_whl=$2
	local pytorch_tag=$3
	
	local vision_version=$4
	local audio_version=$5
	local cuda_arch_list="5.3;6.2;7.2"
	
	if [[ $L4T_RELEASE -ge 34 ]]; then  
		cuda_arch_list="7.2;8.7" # JetPack 5.x (Xavier/Orin)
	fi
	
	echo "building PyTorch $pytorch_whl, torchvision $vision_version, torchaudio $audio_version, cuda arch $cuda_arch_list"

	sh ./scripts/docker_build.sh $pytorch_tag Dockerfile.pytorch \
			--build-arg PYTORCH_URL=$pytorch_url \
			--build-arg PYTORCH_WHL=$pytorch_whl \
			--build-arg TORCHVISION_VERSION=$vision_version \
			--build-arg TORCHAUDIO_VERSION=$audio_version \
			--build-arg TORCH_CUDA_ARCH_LIST=$cuda_arch_list \
			--build-arg OPENCV_URL=$OPENCV_URL \
			--build-arg OPENCV_DEB=$OPENCV_DEB 

	echo "done building PyTorch $pytorch_whl, torchvision $vision_version, torchaudio $audio_version, cuda arch $cuda_arch_list"
}

# PyTorch v1.9.0
build_pytorch "https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl" \
			"torch-1.9.0-cp36-cp36m-linux_aarch64.whl" \
			"l4t-pytorch:r$L4T_VERSION-pth1.9-py3" \
			"v0.10.0" \
			"v0.9.0"


