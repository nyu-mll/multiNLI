#!/bin/bash

# Make sure we have access to HPC-managed libraries.
module load python/intel/2.7.12 

# Run.
PYTHONPATH=. python ebim/ebim.py ebim 
#PYTHONPATH=. python cbow/cbow.py cbow-50
