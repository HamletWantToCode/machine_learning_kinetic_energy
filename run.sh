#!/bin/bash

python=/home/hongbin/anaconda3/envs/project-qml/bin/python
mpirun=/home/hongbin/anaconda3/bin/mpirun
date > out

cd tools/
(time nohup $mpirun -n 4 $python database.py &) >>../out 2>&1

