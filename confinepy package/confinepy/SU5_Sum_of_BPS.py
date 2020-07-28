# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:47:41 2020

@author: samue
"""
import numpy as np
import matplotlib.pyplot as plt
from confinepy.source import solve_BPS, get_BPS_numeric_energy, Sigma_Critical,\
    get_BPS_theoretic_energy

def sum_kink_energy(N,monodromy_arg,kink_separation):
    #Since we are summing 2 solutions, need to add half of total distance,
    #during which the solution is constant
    kink_bd_distance = kink_separation/2 + 40 
    h=0.1
    num=801 # twice usual distance = 80 units
    #continue_kw="default" #don't stop just because BPS energy match
    continue_kw="BPS_Energy"
    #get 2 basic BPS
    x_01, z_linspace, error = solve_BPS(N=N,vac0_arg="x0",vacf_arg="x1",
                                          num=num,h=h,tol=1e-9,sor=1.5,top=True,
                                          kink_bd_distance=kink_bd_distance,
                                          kw="special kink customized center",
                                          continue_kw=continue_kw)
    x_1wk, z_linspace, error = solve_BPS(N=N,vac0_arg="x1",
                                           vacf_arg=monodromy_arg,
                                           num=num,h=h,tol=1e-9,sor=1.5,
                                           top=False,
                                           kink_bd_distance=kink_bd_distance,
                                           kw="special kink customized center",
                                           continue_kw=continue_kw)
    #add 2 fields
    #need to subtract an overall ocnstant of x1
    x_sum = x_01 + x_1wk - Sigma_Critical(N,"x1").imaginary_vector
    #compute energy
    energy = get_BPS_numeric_energy(N=N,x=x_sum,z=z_linspace,h=h)
    
    #plot each components
    phi = np.real(x_sum)
    sigma = np.imag(x_sum)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    for i in range(N-1):
        ax1.plot(z_linspace,phi[:,i],label=r"$\phi_{}$".format(str(i+1)))
    ax1.legend()
    ax2 = fig.add_subplot(122)
    for i in range(N-1):
        ax2.plot(z_linspace,sigma[:,i],label=r"$\sigma_{}$".format(str(i+1)))
    ax2.legend()
    fig.text(x=0,y=0.05,s= "a = {}, E = {}".format(str(a),str(energy)),size=16)
    fig.suptitle("SU({}) Sum of BPS from {} to {} to {}, a={}".format(
        str(N),"x0","x1",monodromy_arg,str(a)),size=20)
    fig.savefig("SU({}) Sum of BPS from {} to {} to {}, a={}.png".format(
        str(N),"x0","x1",monodromy_arg,str(a)), dpi=300)
    
    return energy

#reference 2 BPS energy
theoretic_BPS_energy = get_BPS_theoretic_energy(5,Sigma_Critical(5,"x0"),
                                                Sigma_Critical(5,"x1"))
two_theoretic_energy = 2*theoretic_BPS_energy

a_list = np.arange(1,20)
a_array = np.array(a_list)
energy_list_1 = []
for a in a_list:
    energy = sum_kink_energy(N=5,monodromy_arg = "w1",kink_separation = a)    
    energy_list_1.append(energy)
w1_energy_array = np.array(energy_list_1)
plt.figure()
plt.scatter(a_array,w1_energy_array)
plt.hlines(two_theoretic_energy,a_list[0],a_list[-1],label=r"$2 E_{BPS 1}$")
plt.legend()
plt.title("SU(5) w1 Energy vs Kink Separation")
plt.savefig("SU(5)_w1_Energy_vs_Kink_Separation.png")

a_list = np.arange(1,20)
a_array = np.array(a_list)
energy_list_2 = []
for a in a_list:
    energy = sum_kink_energy(N=5,monodromy_arg = "w2",kink_separation = a)    
    energy_list_2.append(energy)
w2_energy_array = np.array(energy_list_2)
plt.figure()
plt.scatter(a_array,w2_energy_array)
plt.hlines(two_theoretic_energy,a_list[0],a_list[-1],label=r"$2 E_{BPS 1}$")
plt.legend()
plt.title("SU(5) w2 Energy vs Kink Separation")
plt.savefig("SU(5)_w2_Energy_vs_Kink_Separation.png")

plt.figure()
plt.scatter(a_array,w2_energy_array,label="w2")
plt.scatter(a_array,w1_energy_array,label="w1")
plt.hlines(two_theoretic_energy,a_list[0]-1,a_list[-1]+1,label=r"$2 E_{BPS 1}$",linestyles="--")
plt.xlabel("Kink Separation")
plt.ylabel("Energy")
plt.legend()
plt.title("SU(5) Energy vs Kink Separation for Sum of BPS")
plt.savefig("SU(5)_Energy_vs_Kink_Separation_for_Sum_of_BPS.png")

