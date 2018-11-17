#!/bin/bash

set -e

# Replace docker args with the script args
args="$(echo $@ | sed 's/^.*-- //')"
echo "Args: $args" >&2
set -- $args

arg="$1"
if [ -z "$arg" ]; then
    echo 'No run config passed. Can either be "model", "jupyter", "lab", or "notebook".'
    exit 1
fi

set -- ${@:2}

# Activate virtual env
source venv/bin/activate

if ./setup/check_venv.sh
then
    echo "Not in venv, please activate it first."
    exit 1
fi

# Run jupyter notebook environment
if [ "$arg" == 'jupyter' ]; then
    exec jupyter notebook --allow-root $@
fi

# Run jupyter notebook environment
if [ "$arg" == 'lab' ]; then
    exec jupyter lab --allow-root $@
fi

# Run jupyter notebooks directly
if [ "$arg" == 'notebook' ]; then
    exec jupyter nbconvert --to notebook --stdout --execute $@
fi

# Run a specific model + flags + TensorBoard
if [ "$arg" == 'model' ]; then
    tensorboard --logdir data_out >/dev/null & # Run tensorboard in the background
    exec python -m docker.run_config $@ # Pass arg and all successive arguments
fi

echo "Run config $arg unknown!"' Can either be "model", "jupyter", "lab", or "notebook".'
exit 1
