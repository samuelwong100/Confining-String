# -*- coding: utf-8 -*-
"""
File Name: plot_BPS.py
Purpose: plotting
Author: Samuel Wong
"""
import sys
sys.path.append("../Tools")
import numpy as np
import matplotlib.pyplot as plt
from Math import derivative_sample

def plot_BPS(N,z,f,B,h,folder):
    phi = []
    sigma = []
    for i in range(N-1):
        phi.append(np.real(f[:,i]))
        sigma.append(np.imag(f[:,i]))
    
    #get the theoretical derivative for comparison
    dx_theoretic = B.dx(f)
    dphi_theoretic = []
    dsigma_theoretic = []
    for i in range(N-1):
        dphi_theoretic.append(np.real(dx_theoretic[:,i]))
        dsigma_theoretic.append(np.imag(dx_theoretic[:,i]))
        
    #get numerical derivative
    dx_numeric = derivative_sample(f,h)
    dphi_numeric = []
    dsigma_numeric = []
    for i in range(N-1):
        dphi_numeric.append(np.real(dx_numeric[:,i]))
        dsigma_numeric.append(np.imag(dx_numeric[:,i]))
    
    fig = plt.figure(figsize=(14,10))
    ax1 = fig.add_subplot(221)
    for i in range(N-1):
        ax1.plot(z,phi[i],label="$\phi_"+str(i+1)+"$")
    ax1.legend()
    
    ax2 = fig.add_subplot(222)
    for i in range(N-1):
        ax2.plot(z,sigma[i],label="$\sigma_"+str(i+1)+"$")
    ax2.legend()
    
    ax3 = fig.add_subplot(223)
    for i in range(N-1):
        ax3.plot(z, dphi_theoretic[i], '--',label="$d\phi_"+str(i+1)+"theoretic$")
        ax3.plot(z, dphi_numeric[i],label="$d\phi_"+str(i+1)+"numeric$")
    ax3.legend()
    ax3.legend(bbox_to_anchor=(1, 1))
    
    ax4 = fig.add_subplot(224)
    for i in range(N-1):
        ax4.plot(z, dsigma_theoretic[i], '--',label="$d\sigma_"+str(i+1)+"theoretic$")
        ax4.plot(z, dsigma_numeric[i],label="$d\sigma_"+str(i+1)+"numeric$")
    ax4.legend()
    ax4.legend(bbox_to_anchor=(1, 1))
    
    fig.subplots_adjust(wspace=0.7)
    
    fig.savefig(folder+"BPS.png", dpi=300)
