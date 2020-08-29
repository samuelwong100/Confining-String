# -*- coding: utf-8 -*-
###################
# Imports
###################
import sys
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
# epsilon for quantum correction from the command line
epsilon = int(sys.argv[4])

#get appropriate size of grid
L,w = get_canonical_Lw(N,R,epsilon)
sol = confining_string_solver(N=N,charge_arg="w"+str(k),bound_arg="x1",L=L,
                              w=w,R=R,check_point_limit=500)

