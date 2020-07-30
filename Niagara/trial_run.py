# -*- coding: utf-8 -*-
###################
# Imports
###################
import os
import sys
import numpy as np

from confinepy import confining_string_solver, get_canonical_Lw

###################
# Main
###################

# Degree of SU(N) from the command line
#sys.argv[0] always name of script
N = int(sys.argv[1])
# Weight of the quark (corresponding to N-ality)
k = int(sys.argv[2])
# Quark-antiquark separation from the command line
R = int(sys.argv[3])

#get appropriate size of grid
L,w = get_canonical_Lw(R)
#solve
sol = confining_string_solver(N=N,charge_arg="w"+str(k),bound_arg="x1",L,w,R=R,
                              check_point_limit=200)
#exit if job is finished
sys.exit(0)


