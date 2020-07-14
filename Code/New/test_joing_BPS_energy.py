# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:11:09 2020

@author: samue
"""
import numpy as np
import matplotlib.pyplot as plt
from Source import solve_BPS, get_BPS_numeric_energy

def joined_kink_energy(N,monodromy_arg,a):
    kink_separation=a
    kink_bd_distance = kink_separation/2
    h=0.1
    #get 2 basic BPS
    x_01, z_linspace_1, error = solve_BPS(N=N,vac0_arg="x0",vacf_arg="x1",num=401,
                                          h=h,tol=1e-9,sor=1.5,top=True,
                                          kink_bd_distance=kink_bd_distance,
                                          kw="special kink customized center")
    x_1w1, z_linspace_2, error = solve_BPS(N=N,vac0_arg="x1",vacf_arg=monodromy_arg,
                                           num=401,h=h,tol=1e-9,sor=1.5,top=False,
                                           kink_bd_distance=kink_bd_distance,
                                           kw="special kink customized center")
    #join fields
    x_joined = np.concatenate((x_01,x_1w1))
    #join z_linspace
    z_linspace_2_new = z_linspace_2+ 2*z_linspace_2[-1]
    z_joined = np.concatenate((z_linspace_1,z_linspace_2_new))
    z_joined -= z_joined[0] #shift
    
    #compute energy
    energy = get_BPS_numeric_energy(N=N,x=x_joined,z=z_joined,h=h)
    
    #plot each components
    phi = np.real(x_joined)
    sigma = np.imag(x_joined)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    for i in range(N-1):
        ax1.plot(z_joined,phi[:,i],label=r"$\phi_{}$".format(str(i+1)))
    ax1.legend()
    ax2 = fig.add_subplot(122)
    for i in range(N-1):
        ax2.plot(z_joined,sigma[:,i],label=r"$\sigma_{}$".format(str(i+1)))
    ax2.legend()
    fig.text(x=0,y=0.05,s= "a = {}, E = {}".format(str(a),str(energy)),size=16)
    fig.suptitle("SU({}) Joined BPS from {} to {} to {}, a={}".format(
        str(N),"x0","x1",monodromy_arg,str(a)),size=20)
    fig.savefig("SU({}) Joined BPS from {} to {} to {}, a={}.png".format(
        str(N),"x0","x1",monodromy_arg,str(a)), dpi=300)
    
    return energy

a_list = [5,10,20,30,40,50,60,70]
a_array = np.array(a_list)
energy_list = []
for a in a_list:
    energy = joined_kink_energy(N=5,monodromy_arg = "w1",a = a)    
    energy_list.append(energy)
energy_array = np.array(energy_list)
plt.figure()
plt.scatter(a_array,energy_array)
plt.ylim(11.745,11.760)
plt.title("SU(5) w1 Energy vs Kink Separation")
plt.savefig("SU(5)_w1_Energy_vs_Kink_Separation.png")

a_list = [5,10,20,30,40,50,60,70]
a_array = np.array(a_list)
energy_list = []
for a in a_list:
    energy = joined_kink_energy(N=5,monodromy_arg = "w2",a = a)    
    energy_list.append(energy)
energy_array = np.array(energy_list)
plt.figure()
plt.scatter(a_array,energy_array)
plt.ylim(11.745,11.760)
plt.savefig("SU(5) w2 Energy vs Kink Separation.png")


    
