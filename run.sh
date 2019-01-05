#!/bin/bash

python=/home/hongbin/anaconda3/envs/project-qml/bin/python
mpirun=/home/hongbin/anaconda3/envs/project-qml/bin/mpirun
date > out

cd main/
(time nohup $mpirun -n 4 $python database.py &) >>../out 2>&1

