# -*- coding: utf-8 -*-
"""
File Name: plot_tools.py
Purpose: plotting
Author: Samuel Wong
"""
import numpy as np
import matplotlib.pyplot as plt
from equations_tools import Superpotential, derivative_sample

def plot_SU3(z,f,B,num,tol,z0,zf,h,vac0_indx,vacf_indx,ratio=True):
    phi_1 = np.real(f[:,0])
    phi_2 = np.real(f[:,1])
    sigma_1 = np.imag(f[:,0])
    sigma_2 = np.imag(f[:,1])
    
    #get the theoretical derivative for comparison
    dx_theoretic = B.dx(f)
    dphi_1_theoretic = np.real(dx_theoretic[:,0])
    dphi_2_theoretic = np.real(dx_theoretic[:,1])
    dsigma_1_theoretic = np.imag(dx_theoretic[:,0])
    dsigma_2_theoretic = np.imag(dx_theoretic[:,1])
    
    #dx_numeric = (np.roll(f,-1,axis=0) - np.roll(f,1,axis=0))/(2*h)
    dx_numeric = derivative_sample(f,h)
    dphi_1_numeric = np.real(dx_numeric[:,0])
    dphi_2_numeric = np.real(dx_numeric[:,1])
    dsigma_1_numeric = np.imag(dx_numeric[:,0])
    dsigma_2_numeric = np.imag(dx_numeric[:,1])
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221)
    ax1.plot(z, phi_1, label = "$\phi_1$", c='blue')
    ax1.plot(z, phi_2, label = "$\phi_2$", c='green')
    if ratio:
        ax1.plot(z, np.sqrt(3)*phi_1, label = "$\sqrt{3} \phi_1$", c='green')
    ax1.legend()
    
    ax2 = fig.add_subplot(222)
    ax2.plot(z, sigma_1, label = "$\sigma_1$", c='orange')
    ax2.plot(z ,sigma_2, label = "$\sigma_2$", c='red')
    if ratio:
        ax2.plot(z , np.sqrt(3)*sigma_1, label = "$ \sqrt{3} \sigma_1$", c='red')
    ax2.legend()
    
    ax3 = fig.add_subplot(223)
    ax3.plot(z, dphi_1_theoretic, '--',label = "$d\phi_1 theoretic$",
             c='blue')
    ax3.plot(z, dphi_1_numeric, label= "$d\phi_1 numeric$", c='blue')
    ax3.plot(z, dphi_2_theoretic, '--',label = "$d\phi_2 theoretic$",
             c='green')
    ax3.plot(z, dphi_2_numeric,label = "$d\phi_2 numeric$", c='green')
    ax3.legend()
    
    ax4 = fig.add_subplot(224)
    ax4.plot(z, dsigma_1_theoretic, '--',label = "$d\sigma_1 theoretic$",
             c='orange')
    ax4.plot(z, dsigma_1_numeric,label = "$d\sigma_1 numeric$", c='orange')
    ax4.plot(z, dsigma_2_theoretic, '--',label = "$d\sigma_2 theoretic$",
             c='red')
    ax4.plot(z, dsigma_2_numeric,label = "$d\sigma_2 numeric$", c='red')
    ax4.legend()
    
    string1 = "SU(3) BPS Solitons (k="+str(vac0_indx)+" to k=" + str(vacf_indx)+")"
#    fig.suptitle(string1,size=30)
#    string2 = "num=" + str(num) + ", tol=" + str(tol) + ", range=" + "("+ str(z0) \
#                +"," + str(zf) +")" 
#    fig.text(.5,.9,string2, fontsize=18, ha='center')
    fig.savefig('result/'+string1+'.pdf')
    fig.savefig('result/'+string1+'.png', dpi=300)
    
def plot_SU4(z,f,B,num,tol,z0,zf,h,vac0_indx,vacf_indx,x_analytic=None):
    phi_1 = np.real(f[:,0])
    phi_2 = np.real(f[:,1])
    phi_3 = np.real(f[:,2])
    sigma_1 = np.imag(f[:,0])
    sigma_2 = np.imag(f[:,1])
    sigma_3 = np.imag(f[:,2])
    
    #get the theoretical derivative for comparison
    dx_theoretic = B.dx(f)
    dphi_1_theoretic = np.real(dx_theoretic[:,0])
    dphi_2_theoretic = np.real(dx_theoretic[:,1])
    dphi_3_theoretic = np.real(dx_theoretic[:,2])
    dsigma_1_theoretic = np.imag(dx_theoretic[:,0])
    dsigma_2_theoretic = np.imag(dx_theoretic[:,1])
    dsigma_3_theoretic = np.imag(dx_theoretic[:,2])
    
    dx_numeric = derivative_sample(f,h)
    dphi_1_numeric = np.real(dx_numeric[:,0])
    dphi_2_numeric = np.real(dx_numeric[:,1])
    dphi_3_numeric = np.real(dx_numeric[:,2])
    dsigma_1_numeric = np.imag(dx_numeric[:,0])
    dsigma_2_numeric = np.imag(dx_numeric[:,1])
    dsigma_3_numeric = np.imag(dx_numeric[:,2])
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221)
    ax1.plot(z, phi_1, label = "$\phi_1$", c='blue')
    ax1.plot(z, phi_2, label = "$\phi_2$", c='green')
    ax1.plot(z, phi_3, label = "$\phi_3$", c='purple')   
    # plot analytic solution if given
    if x_analytic is not None:
        phi_1_analytic = np.real(x_analytic[:,0])
        phi_2_analytic = np.real(x_analytic[:,1])
        phi_3_analytic = np.real(x_analytic[:,2])
        sigma_1_analytic = np.imag(x_analytic[:,0])
        sigma_2_analytic = np.imag(x_analytic[:,1])
        sigma_3_analytic = np.imag(x_analytic[:,2])
        
        ax1.plot(z, phi_1_analytic, label = "$\phi_1$ analytic", c='blue')
        ax1.plot(z, phi_2_analytic, label = "$\phi_2$ analytic", c='green')
        ax1.plot(z, phi_3_analytic, label = "$\phi_3$ analytic", c='purple')
    ax1.legend()
    
    ax2 = fig.add_subplot(222)
    ax2.plot(z, sigma_1, label = "$\sigma_1$", c='orange')
    ax2.plot(z ,sigma_2, label = "$\sigma_2$", c='red')
    ax2.plot(z ,sigma_3, label = "$\sigma_3$", c='gold')
    if x_analytic is not None:
        ax2.plot(z, sigma_1_analytic, label = "$\sigma_1$ analytic", c='orange')
        ax2.plot(z ,sigma_2_analytic, label = "$\sigma_2$ analytic", c='red')
        ax2.plot(z ,sigma_3_analytic, label = "$\sigma_3$ analytic", c='gold')
    ax2.legend()
    
    ax3 = fig.add_subplot(223)
    ax3.plot(z, dphi_1_theoretic, '--',label = "$d\phi_1 theoretic$",
             c='blue')
    ax3.plot(z, dphi_1_numeric, label= "$d\phi_1 numeric$", c='blue')
    ax3.plot(z, dphi_2_theoretic, '--',label = "$d\phi_2 theoretic$",
             c='green')
    ax3.plot(z, dphi_2_numeric,label = "$d\phi_2 numeric$", c='green')
    ax3.legend()
    ax3.plot(z, dphi_3_theoretic, '--',label = "$d\phi_3 theoretic$",
             c='purple')
    ax3.plot(z, dphi_3_numeric,label = "$d\phi_3 numeric$", c='purple')
    ax3.legend()
    
    ax4 = fig.add_subplot(224)
    ax4.plot(z, dsigma_1_theoretic, '--',label = "$d\sigma_1 theoretic$",
             c='orange')
    ax4.plot(z, dsigma_1_numeric,label = "$d\sigma_1 numeric$", c='orange')
    ax4.plot(z, dsigma_2_theoretic, '--',label = "$d\sigma_2 theoretic$",
             c='red')
    ax4.plot(z, dsigma_2_numeric,label = "$d\sigma_2 numeric$", c='red')
    ax4.plot(z, dsigma_3_theoretic, '--',label = "$d\sigma_3 theoretic$",
             c='gold')
    ax4.plot(z, dsigma_3_numeric,label = "$d\sigma_3 numeric$", c='gold')
    ax4.legend()
    
    string1 = "SU(4) BPS Solitons (k="+str(vac0_indx)+" to k=" + str(vacf_indx)+")"
    #fig.suptitle(string1,size=30)
    #string2 = "num=" + str(num) + ", tol=" + str(tol) + ", range=" + "("+ str(z0) \
    #            +"," + str(zf) +")" 
    #fig.text(.5,.9,string2, fontsize=18, ha='center')
    fig.savefig('result/'+string1+'.pdf')
    fig.savefig('result/'+string1+'.png', dpi=300)
    
def plot_SU_N(N,z,f,B,num,tol,z0,zf,h,vac0_indx,vacf_indx):
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
    
    string1 = "SU("+ str(N)+") BPS Solitons (k="+str(vac0_indx)+" to k=" + \
    str(vacf_indx)+")"
    #fig.suptitle(string1,size=30)
    #string2 = "num=" + str(num) + ", tol=" + str(tol) + ", range=" + "("+ str(z0) \
    #            +"," + str(zf) +")" 
    #fig.text(.5,.9,string2, fontsize=18, ha='center')
    fig.savefig('result/'+string1+'.pdf')
    fig.savefig('result/'+string1+'.png', dpi=300)

    
def plot_W_plane(N,x,B,num,tol,z0,zf,vac0_indx,vacf_indx):
    W = Superpotential(N)
    
    plt.figure()
    #plot the line in W-plane
    W_x = W(x)
    W_x = W_x.reshape(W_x.size)
    W_x_real = np.real(W_x)
    W_x_imag = np.imag(W_x)
    plt.plot(W_x_real,W_x_imag,color="black")
    
    #theoretical start and end point
    start = W(B.xmin0)
    end = W(B.xminf)
    plt.plot(np.real(start), np.imag(start), marker='o',
                      markersize=3, color="red",label = "theoretical start")
    plt.plot(np.real(end), np.imag(end), marker='o',
                  markersize=3, color="blue",label = "theoretical end")
    if N == 3:
        plt.xlim(-4,4)
        plt.ylim(-4,4)
    elif N == 4:
        plt.xlim(-6,6)
        plt.ylim(-6,6)
    elif N == 5:
        plt.xlim(-8,8)
        plt.ylim(-8,8)
    else:
        plt.axis("equal")
    plt.xlabel("Re(W)")
    plt.ylabel("Im(W)")
    plt.legend()
    string1 = "SU("+str(N)+") W-Plane (k="+str(vac0_indx)+" to k=" +\
                str(vacf_indx)+")"
    plt.title(string1,fontsize=15,pad =20)
    string2 = "num=" + str(num) + ", tol=" + str(tol) + ", range=" + "("+ str(z0) \
                +"," + str(zf) +")"
    plt.figtext(0.5,0.85,string2, ha='center')
    plt.savefig('result/'+string1+'.pdf')
    plt.savefig('result/'+string1+'.png')
    
    #save the array for W-plane line
    np.savez('result/'+string1, x=W_x_real, y=W_x_imag)
    