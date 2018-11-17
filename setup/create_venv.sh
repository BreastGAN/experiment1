#!/bin/bash

set -e

venv="venv"

mkdir "$venv"
python3 -m pip install virtualenv --user
python3 -m virtualenv "$venv" --no-site-packages

source "$venv"/bin/activate
pip install -r "requirements.txt"

# Install the pre-commit hook
if [ -z "$CI" ]; then
    rm -f .git/hooks/pre-commit
    ln -s ../../setup/pre-commit.sh .git/hooks/pre-commit
fi

# Install jupyter_tensorboard
pip install jupyter_tensorboard==0.1.5

# Install code_prettify
jupyter contrib nbextension install --user
jupyter nbextension enable code_prettify/code_prettify

# Install nbstripout hooks
git config filter.nbstripout.clean 'nbstripout'
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true

git config diff.ipynb.textconv 'nbstripout -t'
