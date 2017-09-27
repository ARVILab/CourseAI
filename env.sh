#!/usr/bin/env bash

apt-get update
apt-get -y install python-pip
apt-get -y install ipython ipython-notebook
apt-get -y install cython

pip2 install --upgrade pip
pip2 install cython
pip2 install numpy
pip2 install opencv-python
pip2 install -U scikit-learn
pip2 install --upgrade tensorflow-gpu
pip2 install keras
pip2 install pandas
pip2 install scipy
pip2 install tsne
pip2 install --upgrade jupyter
pip2 install h5py
pip2 install pillow
pip2 install pylab

python2 -m pip install ipykernel
python2 -m ipykernel install --user
