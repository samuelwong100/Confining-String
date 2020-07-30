# -*- coding: utf-8 -*-
###################
# Imports
###################
import os
import sys
import numpy as np

from confinepy import confining_string_solver

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

# # Name of the output file and checkpoint file
# savefile = 'SU{}_k{}_R{}.npz'.format(N, k, R)

# # Output directory and file
# savedir = os.path.expandvars('$SCRATCH/confinement/SU{}/k{}'.format(N, k))
# filename = os.path.join(savedir, savefile)

# Checkpoint directory
#checkpoint_dir = os.path.expandvars('$SCRATCH/checkpoints/confinement')
#checkpoint_dir = os.path.join(checkpoint_dir, 'SU{}/k{}/R{}'.format(N, k, R))

# # Files containing checkpoint information
# checkpoint_file = os.path.join(checkpoint_dir, savefile)
# info_file = os.path.join(checkpoint_dir, 'checkpoint.txt')

# # Exit if the job has already been completed
# if os.path.exists(filename):
#     sys.exit(0)

# # 1D fields to contain the initial conditions
# f1 = Field1D(N - 1, -15, 0, 0.1)
# f2 = Field1D(N - 1, 0, 15, 0.1)

# # Set up the desired boundary values
# f1.field[:, :7 * f1.nz // 8] = 2j * np.pi * rho[:, np.newaxis] / N
# f1.field[:, 7 * f1.nz // 8:] = 2j * np.pi * w[k - 1][:, np.newaxis]

# f2.field[:, :f2.nz // 8] = 0
# f2.field[:, f2.nz // 8:] = 2j * np.pi * rho[:, np.newaxis] / N

# # Relax to find the initial conditions
# s1 = RelaxationSolver1D(f1, func=W.bps_eom)
# s1.solve(tol=1e-9, maxiter=10000, omega=1.5)

# s2 = RelaxationSolver1D(f2, func=W.bps_eom)
# s2.solve(tol=1e-9, maxiter=10000, omega=1.5)

# # Function to set the initial conditions of the 2D field
# def set_initial_conditions(field, separation):
#     field.field[:, :, :field.ny // 2] = f1.field[:, np.newaxis, :field.ny // 2]
#     field.field[:, :, field.ny // 2:] = f2.field[:, np.newaxis, :]
#     for i in range(field.nz):
#         if np.abs(field.z[i]) > separation / 2:
#             field.field[:, i, :] = 2j * np.pi * rho[:, np.newaxis] / N

# # Set the left/right boundary such that there are at least 5 units of space
# if R % 10 == 0:
#     zmax = max(15, R // 2 + 5)
# else:
#     zmax = max(15, (R // 10 + 2) * 5)

# # Meson with weight w_k
# m = Meson(-R / 2, R / 2, w[k - 1])

# # Create the checkpoint directory
# os.makedirs(checkpoint_dir, exist_ok=True)

# # Try to read the checkpoint
# #tries to load checkpoint file, if don't exists, new run is true (first time)
# new_run = True
# if os.path.exists(checkpoint_file):
#     try:
#         f = Field2D.load(checkpoint_file)
#         with open(info_file, 'r') as info:
#             iterations = int(info.readline())
#             error = float(info.readline())
#     except:
#         pass
#     else:
#         new_run = False

# # Create a new field if the checkpoint could not be read
# if new_run:
#     f = Field2D(N - 1, -zmax, zmax, -15, 15, 0.1)
#     set_initial_conditions(f, R)
#     iterations = 0
#     error = np.inf

# # Set up the RelaxationSolver
# s = RelaxationSolver2D(f, W.eom, constant=m.eom(f))

# # Maximum iterations and error tolerance
# maxiter = 500000 - iterations
# tol = 1e-9

# # Loop until the error tolerance is surpassed or maxiter is reached
# while error > tol and maxiter > 0:

#     # Perform a maximum of 1000 iterations before checkpointing
#     if maxiter > 1000:
#         nextiter = 1000
#     else:
#         nextiter = maxiter

#     # Relax the field and then update the iteration count
#     thisiter, error = s.symmetric_solve(maxiter=nextiter, tol=tol, omega=1.5)
#     iterations += thisiter
#     maxiter -= thisiter

#     # Update the checkpoint file
#     f.save(checkpoint_file)
#     with open(info_file, 'w') as info:
#         info.write(str(iterations) + '\n' + str(error) + '\n')

# # Make the directory for saving
# os.makedirs(savedir, exist_ok=True)

# # Save the field
# if not os.path.exists(filename):
#     f.save(filename)

