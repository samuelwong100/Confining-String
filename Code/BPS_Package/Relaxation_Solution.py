# -*- coding: utf-8 -*-
"""
File Name: Relaxation_Solution.py
Purpose: Function that calls relaxation and apply it to BPS equations
Author: Samuel Wong
"""
import numpy as np
from Relaxation import Relaxation
from equations_tools import BPS
from plot_tools import plot_SU3, plot_SU4, plot_SU_N, plot_W_plane
from Energy import Energy

def SU_N_Relaxation_Solution(N,vac0_indx,vacf_indx,tol,num,z0,zf,recalculate=False,
                            ratio=True,x_analytic=None,f0="special kink"):
    #pixel
    h = (zf-z0)/num
    
    title = "result/SU(" + str(N) + ") BPS Solitons (k="+str(vac0_indx)+" to k=" +\
        str(vacf_indx)+")"+", num=" + str(num) + ", tol=" + str(tol) + \
        ", range=" + "("+ str(z0)  +"," + str(zf) +")"
        
    #To save time, we can decide whether to recalculate or plot old results here.
    if recalculate:
        # define BPS object, which contains BPS equations and bounds
        B = BPS(N,vac0_indx,vacf_indx)
        # define the relaxation object; call the second order ODE from BPS
        R = Relaxation(B.ddx,z0,zf,B.xmin0[0],B.xminf[0])
        # solve with specified number of points on grid
        R.solve(num,tol=tol,f0=f0,diagnose=True)
        #save result
        np.savez(title,z=R.sol_z, x=R.sol_f)
        #rename variables
        z = R.sol_z
        x = R.sol_f
    else:
        data=np.load(title+'.npz')
        B = BPS(N,vac0_indx,vacf_indx)
        #rename variables
        z = data['z']
        x = data['x']
        
    # plot
    if N == 3:
        plot_SU3(z,x,B,num,tol,z0,zf,h,vac0_indx,vacf_indx,ratio)
    elif N == 4:
        plot_SU4(z,x,B,num,tol,z0,zf,h,vac0_indx,vacf_indx,x_analytic)
    else:
        plot_SU_N(N,z,x,B,num,tol,z0,zf,h,vac0_indx,vacf_indx)
        
    #w-plane and energy
    plot_W_plane(N,x,B,num,tol,z0,zf,vac0_indx,vacf_indx)
    theoretic_energy,numeric_energy = Energy(N,num,vac0_indx,vacf_indx,x,z,h)
    print("theoretical energy =",theoretic_energy)
    print("numerical energy =", numeric_energy)
        
    return z,x

def SU3_Relaxation_Solution(vac0_indx,vacf_indx,tol,num,z0,zf,recalculate=False,
                            ratio=True):
    N = 3
    #pixel
    h = (zf-z0)/num
    
    title = "result/SU(" + str(N) + ") BPS Solitons (k="+str(vac0_indx)+" to k=" +\
        str(vacf_indx)+")"+", num=" + str(num) + ", tol=" + str(tol) + \
        ", range=" + "("+ str(z0)  +"," + str(zf) +")"
    
    #To save time, we can decide whether to recalculate or plot old results here.
    if recalculate:
        # define BPS object, which contains BPS equations and bounds
        B = BPS(N,vac0_indx,vacf_indx)
        # define the relaxation object; call the second order ODE from BPS
        R = Relaxation(B.ddx,z0,zf,B.xmin0[0],B.xminf[0])
        # solve with specified number of points on grid
        R.solve(num,tol=tol,f0="special kink",diagnose=True)
        #save result
        np.savez(title,z=R.sol_z, x=R.sol_f)
        plot_SU3(R.sol_z,R.sol_f,B,num,tol,z0,zf,h,vac0_indx,vacf_indx,ratio)
        #rename variables
        z = R.sol_z
        x = R.sol_f
    else:
        data=np.load(title+'.npz')
        B = BPS(N,vac0_indx,vacf_indx)
        plot_SU3(data['z'],data['x'],B,num,tol,z0,zf,h,vac0_indx,vacf_indx,ratio)
        #rename variables
        z = data['z']
        x = data['x']
        
    plot_W_plane(N,x,B,num,tol,z0,zf,vac0_indx,vacf_indx)
    theoretic_energy,numeric_energy = Energy(N,num,vac0_indx,vacf_indx,x,z,h)
    print("theoretical energy =",theoretic_energy)
    print("numerical energy =", numeric_energy)

def SU4_Relaxation_Solution(vac0_indx,vacf_indx,tol,num,z0,zf,recalculate=False):
    N = 4
    #pixel
    h = (zf-z0)/num
    
    title = "result/SU(" + str(N) + ") BPS Solitons (k="+str(vac0_indx)+" to k=" +\
        str(vacf_indx)+")"+", num=" + str(num) + ", tol=" + str(tol) + \
        ", range=" + "("+ str(z0)  +"," + str(zf) +")"
    
    #To save time, we can decide whether to recalculate or plot old results here.
    if recalculate:
        # define BPS object, which contains BPS equations and bounds
        B = BPS(N,vac0_indx,vacf_indx)
        # define the relaxation object; call the second order ODE from BPS
        R = Relaxation(B.ddx,z0,zf,B.xmin0[0],B.xminf[0])
        # solve with specified number of points on grid
        R.solve(num,tol=tol,f0="special kink",diagnose=True)
        #save result
        np.savez(title,z=R.sol_z, x=R.sol_f)
        plot_SU4(R.sol_z,R.sol_f,B,num,tol,z0,zf,h,vac0_indx,vacf_indx)
        #rename variables
        z = R.sol_z
        x = R.sol_f
    else:
        data=np.load(title+'.npz')
        B = BPS(N,vac0_indx,vacf_indx)
        plot_SU4(data['z'],data['x'],B,num,tol,z0,zf,h,vac0_indx,vacf_indx)
        #rename variables
        z = data['z']
        x = data['x']
        
    plot_W_plane(N,x,B,num,tol,z0,zf,vac0_indx,vacf_indx)
    theoretic_energy,numeric_energy = Energy(N,num,vac0_indx,vacf_indx,x,z,h)
    print("theoretical energy =",theoretic_energy)
    print("numerical energy =", numeric_energy)
    
def SU4_explore(vac0_indx,vacf_indx,tol,num,z0,zf,recalculate=False):
    N = 4
    #pixel
    h = (zf-z0)/num
    
    title = "result/EXPLORE_SU(" + str(N) + ") BPS Solitons (k="+str(vac0_indx)+" to k=" +\
        str(vacf_indx)+")"+", num=" + str(num) + ", tol=" + str(tol) + \
        ", range=" + "("+ str(z0)  +"," + str(zf) +")"
    
    #To save time, we can decide whether to recalculate or plot old results here.
    if recalculate:
        # define BPS object, which contains BPS equations and bounds
        B = BPS(N,vac0_indx,vacf_indx)
        # define the relaxation object; call the second order ODE from BPS
        R = Relaxation(B.ddx,z0,zf,B.xmin0[0],B.xminf[0])
        # solve with specified number of points on grid
        R.solve(num,tol=tol,f0="special kink",diagnose=True)
        #save result
        np.savez(title,z=R.sol_z, x=R.sol_f)
        #rename variables
        z = R.sol_z
        x = R.sol_f
    else:
        data=np.load(title+'.npz')
        B = BPS(N,vac0_indx,vacf_indx)
        #rename variables
        z = data['z']
        x = data['x']
        
    plot_SU4(z,x,B,num,tol,z0,zf,h,vac0_indx,vacf_indx)
    return z,x

if __name__ == "__main__":
    N = 4
    z0 = -20
    zf = 20
    num = 200
    k0="w1"
    kf=1
    tol = 0.1
    
    z,x=SU_N_Relaxation_Solution(N,k0,kf,tol=tol,num=num,z0=z0,
                                        zf=zf,recalculate=True)
    
