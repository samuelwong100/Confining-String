# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#from confinepy.source import solve_BPS
from source import solve_BPS, Sigma_Critical

def streamline(N,monodromy_arg,a,tol_stream):
    kink_separation=a
    # kink_bd_distance = kink_separation/2
    # h=0.1
    # #get 2 basic BPS
    # x_01, z_linspace_1, error = solve_BPS(N=N,vac0_arg="x0",vacf_arg="x1",num=401,
    #                                       h=h,tol=1e-9,sor=1,top=True,
    #                                       kink_bd_distance=kink_bd_distance,
    #                                       kw="special kink customized center",
    #                                       continue_kw="default")
    # x_1w1, z_linspace_2, error = solve_BPS(N=N,vac0_arg="x1",vacf_arg=monodromy_arg,
    #                                        num=401,h=h,tol=1e-9,sor=1,top=False,
    #                                        kink_bd_distance=kink_bd_distance,
    #                                        kw="special kink customized center",
    #                                        continue_kw="default")
    #join fields
    #x_joined = np.concatenate((x_01,x_1w1))
    
    kink_bd_distance = kink_separation/2 + 40 
    h=0.1
    num=401 # twice usual distance = 80 units
    #continue_kw="default" #don't stop just because BPS energy match
    continue_kw="default"
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
    x_joined = x_sum
    
    x_streamline, z_streamline, error = \
        solve_BPS(N=N,vac0_arg="x1",vacf_arg=monodromy_arg,
            num=num,h=h,tol=tol_stream,sor=1,
            kw="x0_given",
            continue_kw="default",x0_given=x_joined,folder="streamline/")

    #plot initial joined fields
    #plot each components
    phi = np.real(x_joined)
    sigma = np.imag(x_joined)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    for i in range(N-1):
        ax1.plot(z_streamline,phi[:,i],label=r"$\phi_{}$".format(str(i+1)))
    ax1.legend()
    ax2 = fig.add_subplot(122)
    for i in range(N-1):
        ax2.plot(z_streamline,sigma[:,i],label=r"$\sigma_{}$".format(str(i+1)))
    ax2.legend()
    fig.suptitle("SU({}) Initial Joined BPS from {} to {} to {}, a={}".format(
        str(N),"x0","x1",monodromy_arg,str(a)),size=20)
    fig.savefig("streamline/SU({}) Initial Joined BPS from {} to {} to {}, a={}.png".format(
        str(N),"x0","x1",monodromy_arg,str(a)), dpi=300)
    
    #plot streamline fields
    #plot each components
    phi = np.real(x_streamline)
    sigma = np.imag(x_streamline)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    for i in range(N-1):
        ax1.plot(z_streamline,phi[:,i],label=r"$\phi_{}$".format(str(i+1)))
    ax1.legend()
    ax2 = fig.add_subplot(122)
    for i in range(N-1):
        ax2.plot(z_streamline,sigma[:,i],label=r"$\sigma_{}$".format(str(i+1)))
    ax2.legend()
    fig.suptitle("SU({}) Streamline BPS from {} to {} to {}, a={}".format(
        str(N),"x0","x1",monodromy_arg,str(a)),size=20)
    fig.savefig("streamline/SU({}) Streamline BPS from {} to {} to {}, a={}.png".format(
        str(N),"x0","x1",monodromy_arg,str(a)), dpi=300)



