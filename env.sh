#!/usr/bin/env bash

export LC_ALL=C

apt-get update
apt-get -y install python-pip
apt-get -y install ipython ipython-notebook
apt-get -y install cython
apt-get -y install python-numpy python-scipy python-matplotlib
apt-get -y install build-essential autoconf libtool pkg-config python-opengl
apt-get -y install python-imaging
apt-get -y install python-dev
apt-get -y install libopenblas-dev libblas-dev libatlas-base-dev

pip2 install --upgrade pip
pip2 install cython
pip2 install --upgrade jupyter
pip2 install numpy
pip2 install opencv-python
pip2 install -U scikit-learn
pip2 install --upgrade tensorflow-gpu
pip2 install keras
pip2 install pandas
pip2 install scipy
pip2 install h5py
pip2 install pillow, pylab, tsne, annoy

python2 -m pip install ipykernel
python2 -m ipykernel install --user
