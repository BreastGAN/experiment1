#!/bin/bash
img="breastgan/experiment1:gpu-latest"

set -e

pass="$1"
if [ -z $pass ]; then
    echo "Please supply a Jupyter notebook password as the first parameter."
    exit 1
fi

set -- "${@:2}"

# GPU
cmd="sudo docker run --runtime=nvidia -it -e PASSWORD=$pass -p 8888:8888 -p 6006:6006 $img -- $@"
# CPU
#cmd="sudo docker run -it -e PASSWORD=$pass -p 8888:8888 -p 6006:6006 $img -- $@"
echo "Running: $cmd"
eval "$cmd"
