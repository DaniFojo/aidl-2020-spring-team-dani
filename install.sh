#!/usr/bin/env bash

#Basic stuff
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update
    sudo apt install build-essential libssl-dev libffi-dev python-dev libopenmpi-dev python3-tk
    sudo apt install python3-pip
    sudo pip3 install virtualenv 
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Mac OSX brew stuff"
elif [[ "$OSTYPE" == "WINDOWS"* ]]; then
    echo "# No idea, maybe choco something?"
fi


#Create environment
python3 -m venv env
. env/bin/activate

# pyton packages
pip3 install pylint autopep8 matplotlib gym torch torchvision numpy tensorboard pystan Box2D pyglet==1.2.4 gym[atari] pybullet scikit-image

