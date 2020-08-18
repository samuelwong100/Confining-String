# -*- coding: utf-8 -*-
###################
# Imports
###################
import sys
from confinepy import confining_string_solver

###################
# Main
###################

#sys.argv[0] always name of script
# Quark-antiquark separation from the command line
R = int(sys.argv[1])
# Dimension of square grid from the command line
L = int(sys.argv[2])
w = L


sol = confining_string_solver(N=3,charge_arg="w1",bound_arg="x1",L=L,
                              w=L,R=R,check_point_limit=500)
