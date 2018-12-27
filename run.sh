#!/bin/bash

python=/anaconda3/envs/workspace/bin/python
mpirun=/anaconda3/envs/workspace/bin/mpirun
date > out

cd tools/
(time nohup $mpirun -n 4 $python database.py &) >>../out 2>&1

