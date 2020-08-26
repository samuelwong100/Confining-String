# -*- coding: utf-8 -*-
"""
File Name: extract_tension_function.py
Purpose: extract the dependence of tension on N and p
Author: Samuel Wong
"""
import numpy as np
import matplotlib.pyplot as plt

def get_tension_dict():
    #load data
    data = np.load("tension.npz")
    tensions = data['arr_0']
    #create a tension dictionary
    tension_dict = {}
    for point in tensions:
        N,p,T,dT = point
        tension_dict[(N,p)] = (T,dT)
    return tension_dict

#functions for tension and tension uncertainty
def T(N,p):
    return tension_dict[(N,p)][0]

def dT(N,p):
    return tension_dict[(N,p)][1]

def get_f_df_dict():
    #create dictionary for f function
    f_dict = {}
    df_dict = {}
    for arg in tension_dict:
        N,p = arg
        f_dict[(N,p)] = T(N,p)/T(N,1)
        df_dict[(N,p)] = (T(N,p)/T(N,1))* np.sqrt((dT(N,p)/T(N,p))**2 + \
                (dT(N,1)/T(N,1))**2)
    return f_dict, df_dict

#create f function
def f(N,p):
    return f_dict[(N,p)]

def df(N,p):
    return df_dict[(N,p)]

def get_T_dT_f_df_by_p(p):
    Np_list = np.arange(2*p,10+1)
    Tp_list = np.array([T(i,p) for i in Np_list])
    dTp_list = np.array([dT(i,p) for i in Np_list])
    fp_list = np.array([f(i,p) for i in Np_list])
    dfp_list = np.array([df(i,p) for i in Np_list])
    return Np_list, Tp_list, dTp_list, fp_list, dfp_list
    
def sine_law(p,N_list):
    return np.sin(p*np.pi/N_list)/np.sin(np.pi/N_list)

def sine_law2(p_ls,N):
    return np.sin(p_ls*np.pi/N)/np.sin(np.pi/N)

def casimir(p,N_list):
    return p*(N_list-p)/(N_list - 1)

def casimir2(p_ls,N):
    return p_ls*(N-p_ls)/(N - 1)

def plot_tensions():
    #plot tensions as a function of N with different N-ality series
    plt.figure()
    plt.errorbar(x=N1_list,y=T1_list,yerr=dT1_list,fmt='o',ls='-',label='k=1')
    plt.errorbar(x=N2_list,y=T2_list,yerr=dT2_list,fmt='o',ls='-',label='k=2')
    plt.errorbar(x=N3_list,y=T3_list,yerr=dT3_list,fmt='o',ls='-',label='k=3')
    plt.errorbar(x=N4_list,y=T4_list,yerr=dT4_list,fmt='o',ls='-',label='k=4')
    plt.errorbar(x=N5_list,y=T5_list,yerr=dT5_list,fmt='o',ls='-',label='k=5')
    plt.xlabel('N')
    plt.ylabel('Tension')
    plt.title('Tension vs N')
    plt.legend()
    plt.savefig('Tension vs N.png')
    plt.show()

def plot_f():
    #plot f as a function of N with different N-ality series
    plt.figure()
    plt.errorbar(x=N2_list,y=f2_list,yerr=df2_list,fmt='o',label='k=2')
    plt.errorbar(x=N3_list,y=f3_list,yerr=df3_list,fmt='o',label='k=3')
    plt.errorbar(x=N4_list,y=f4_list,yerr=df4_list,fmt='o',label='k=4')
    plt.errorbar(x=N5_list,y=f5_list,yerr=df5_list,fmt='o',label='k=5')
    plt.xlabel('N')
    plt.ylabel('f')
    plt.title('f vs N')
    plt.legend()
    plt.savefig('f vs N.png',dpi=500)
    plt.show()
    
def plot_f_compare_sine():
    #plot f as a function of N with different N-ality series
    plt.figure()
    #plt.errorbar(x=N1_list,y=f1_list,yerr=df1_list,fmt='o',ls='-',label='k=1')
    plt.errorbar(x=N2_list,y=f2_list,yerr=df2_list,fmt='o',label='k=2')
    plt.errorbar(x=N3_list,y=f3_list,yerr=df3_list,fmt='o',label='k=3')
    plt.errorbar(x=N4_list,y=f4_list,yerr=df4_list,fmt='o',label='k=4')
    plt.errorbar(x=N5_list,y=f5_list,yerr=df5_list,fmt='o',label='k=5')
    #plot Shenker's sine law for comparison
    plt.scatter(x=N2_list,y=sine_law(2,N2_list),marker='x',label='Sine Law k=2')
    plt.scatter(x=N3_list,y=sine_law(3,N3_list),marker='x',label='Sine Law k=3')
    plt.scatter(x=N4_list,y=sine_law(4,N4_list),marker='x',label='Sine Law k=4')
    plt.scatter(x=N5_list,y=sine_law(5,N5_list),marker='x',label='Sine Law k=5')
    plt.xlabel('N')
    plt.ylabel('f')
    plt.title('f vs N compared with Sine Law')
    plt.legend()
    plt.savefig('f vs N Compare Sine.png')
    plt.show()

def plot_f_compare_sine_side_by_side():
    #plot f as a function of N with different N-ality series
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.errorbar(x=N2_list,y=f2_list,yerr=df2_list,fmt='o',label='k=2')
    plt.errorbar(x=N3_list,y=f3_list,yerr=df3_list,fmt='o',label='k=3')
    plt.errorbar(x=N4_list,y=f4_list,yerr=df4_list,fmt='o',label='k=4')
    plt.errorbar(x=N5_list,y=f5_list,yerr=df5_list,fmt='o',label='k=5')
    plt.xlabel('N')
    plt.ylabel('f')
    plt.legend()
    #plot Shenker's sine law for comparison
    plt.subplot(1,2,2)
    plt.scatter(x=N2_list,y=sine_law(2,N2_list),marker='x',label='Sine Law k=2')
    plt.scatter(x=N3_list,y=sine_law(3,N3_list),marker='x',label='Sine Law k=3')
    plt.scatter(x=N4_list,y=sine_law(4,N4_list),marker='x',label='Sine Law k=4')
    plt.scatter(x=N5_list,y=sine_law(5,N5_list),marker='x',label='Sine Law k=5')
    plt.xlabel('N')
    plt.ylabel('f')
    plt.suptitle('f vs N compared with Sine Law')
    plt.legend()
    plt.savefig('f vs N compared with Sine Law side by side.png')
    plt.show()
    
def plot_f_compare_casimir():
    #plot f as a function of N with different N-ality series
    plt.figure()
    #plt.errorbar(x=N1_list,y=f1_list,yerr=df1_list,fmt='o',ls='-',label='k=1')
    plt.errorbar(x=N2_list,y=f2_list,yerr=df2_list,fmt='o',label='k=2')
    plt.errorbar(x=N3_list,y=f3_list,yerr=df3_list,fmt='o',label='k=3')
    plt.errorbar(x=N4_list,y=f4_list,yerr=df4_list,fmt='o',label='k=4')
    plt.errorbar(x=N5_list,y=f5_list,yerr=df5_list,fmt='o',label='k=5')
    #plot casimir scaling for comparison
    plt.gca().set_prop_cycle(None)
    plt.plot(N2_list,casimir(2,N2_list),marker='x',ls='--',label='Casimir k=2')
    plt.plot(N3_list,casimir(3,N3_list),marker='x',ls='--',label='Casimir k=3')
    plt.plot(N4_list,casimir(4,N4_list),marker='x',ls='--',label='Casimir k=4')
    plt.plot(N5_list,casimir(5,N5_list),marker='x',ls='--',label='Casimir k=5')
    plt.xlabel('N')
    plt.ylabel('f')
    plt.title('f vs N compared with Casimir')
    plt.legend()
    plt.savefig('f vs N Compare Casimir.png')
    plt.show()

def plot_f_compare_sqrt_casimir():
    #plot f as a function of N with different N-ality series
    plt.figure()
    #plt.errorbar(x=N1_list,y=f1_list,yerr=df1_list,fmt='o',ls='-',label='k=1')
    plt.errorbar(x=N2_list,y=f2_list,yerr=df2_list,fmt='o',ls='-',label='k=2')
    plt.errorbar(x=N3_list,y=f3_list,yerr=df3_list,fmt='o',ls='-',label='k=3')
    plt.errorbar(x=N4_list,y=f4_list,yerr=df4_list,fmt='o',ls='-',label='k=4')
    plt.errorbar(x=N5_list,y=f5_list,yerr=df5_list,fmt='o',ls='-',label='k=5')
    #plot sqrt casimir scaling for comparison
    plt.gca().set_prop_cycle(None)
    plt.plot(N2_list,np.sqrt(casimir(2,N2_list)),marker='x',ls='--',label='sqrt Casimir k=2')
    plt.plot(N3_list,np.sqrt(casimir(3,N3_list)),marker='x',ls='--',label='sqrt Casimir k=3')
    plt.plot(N4_list,np.sqrt(casimir(4,N4_list)),marker='x',ls='--',label='sqrt Casimir k=4')
    plt.plot(N5_list,np.sqrt(casimir(5,N5_list)),marker='x',ls='--',label='sqrt Casimir k=5')
    plt.xlabel('N')
    plt.ylabel('f')
    plt.title('f vs N compared with sqrt Casimir')
    plt.legend()
    plt.savefig('f vs N compared with sqrt Casimir.png')
    plt.show()

def plot_f_compare_everything():
    #plot f as a function of N with different N-ality series
    plt.figure()
    #plt.errorbar(x=N1_list,y=f1_list,yerr=df1_list,fmt='o',ls='-',label='k=1')
    plt.errorbar(x=N2_list,y=f2_list,yerr=df2_list,fmt='o',label='k=2')
    plt.errorbar(x=N3_list,y=f3_list,yerr=df3_list,fmt='o',label='k=3')
    plt.errorbar(x=N4_list,y=f4_list,yerr=df4_list,fmt='o',label='k=4')
    plt.errorbar(x=N5_list,y=f5_list,yerr=df5_list,fmt='o',label='k=5')
    #plot casimir scaling for comparison
    # plt.scatter(x=N2_list,y=casimir(2,N2_list),marker='x',label='Casimir k=2')
    # plt.scatter(x=N3_list,y=casimir(3,N3_list),marker='x',label='Casimir k=3')
    # plt.scatter(x=N4_list,y=casimir(4,N4_list),marker='x',label='Casimir k=4')
    # plt.scatter(x=N5_list,y=casimir(5,N5_list),marker='x',label='Casimir k=5')
    #plot casimir scaling for comparison
    plt.scatter(x=N2_list,y=np.sqrt(casimir(2,N2_list)),marker='*',label='sqrt Casimir k=2')
    plt.scatter(x=N3_list,y=np.sqrt(casimir(3,N3_list)),marker='*',label='sqrt Casimir k=3')
    plt.scatter(x=N4_list,y=np.sqrt(casimir(4,N4_list)),marker='*',label='sqrt Casimir k=4')
    plt.scatter(x=N5_list,y=np.sqrt(casimir(5,N5_list)),marker='*',label='sqrt Casimir k=5')
    #plot Shenker's sine law for comparison
    # plt.scatter(x=N2_list,y=sine_law(2,N2_list),marker='^',label='Sine Law k=2')
    # plt.scatter(x=N3_list,y=sine_law(3,N3_list),marker='^',label='Sine Law k=3')
    # plt.scatter(x=N4_list,y=sine_law(4,N4_list),marker='^',label='Sine Law k=4')
    # plt.scatter(x=N5_list,y=sine_law(5,N5_list),marker='^',label='Sine Law k=5')
    plt.xlabel('N')
    plt.ylabel('f')
    plt.title('f vs N compared with 3 Laws')
    plt.legend()
    plt.savefig('f vs N Compared with 3 Laws.png')
    plt.show()
    
def plot_f_compare_casimir_ratio():
    #plot f as a function of N with different N-ality series
    plt.figure()
    #plot casimir scaling for comparison
    plt.scatter(x=N2_list,y=f2_list/casimir(2,N2_list),marker='x',label='Casimir k=2')
    plt.scatter(x=N3_list,y=f3_list/casimir(3,N3_list),marker='x',label='Casimir k=3')
    plt.scatter(x=N4_list,y=f4_list/casimir(4,N4_list),marker='x',label='Casimir k=4')
    plt.scatter(x=N5_list,y=f5_list/casimir(5,N5_list),marker='x',label='Casimir k=5')
    plt.xlabel('N')
    plt.ylabel('f')
    plt.title('f vs N compared with Casimir')
    plt.legend()
    plt.savefig('f vs N Compare Casimir.png')
    plt.show()

def master_dict_array():
    master_dict={}
    master_list= []
    for index in tension_dict:
        master_dict[index] = *tension_dict[index], f_dict[index], df_dict[index]
        N,p = index
        master_list.append((N,p,*tension_dict[index], f_dict[index],
                            df_dict[index]))
    master_array = np.array(master_list)
    return master_dict, master_array

def get_T_dT_f_df_by_N(N):
    pN_ls = np.arange(1,int(N/2)+1)
    TN_ls = np.array([T(N,i) for i in pN_ls])
    dTN_ls = np.array([dT(N,i) for i in pN_ls])
    fN_ls = np.array([f(N,i) for i in pN_ls])
    dfN_ls = np.array([df(N,i) for i in pN_ls])
    return pN_ls, TN_ls, dTN_ls, fN_ls, dfN_ls

def reflect(arr):
    return np.append(arr,np.flip(arr)[1:])

def plot_f_vs_p(N):
    if N == 10:
        #plot f as a function of k with different N series
        plt.figure()
        # plt.errorbar(x=p2_ls,y=f2_ls,yerr=df2_ls,fmt='o',label='N=2')
        # plt.errorbar(x=p3_ls,y=f3_ls,yerr=df3_ls,fmt='o',label='N=3')
        # plt.errorbar(x=p4_ls,y=f4_ls,yerr=df4_ls,fmt='o',label='N=4')
        # plt.errorbar(x=p5_ls,y=f5_ls,yerr=df5_ls,fmt='o',label='N=5')
        # plt.errorbar(x=p6_ls,y=f6_ls,yerr=df6_ls,fmt='o',label='N=6')
        # plt.errorbar(x=p7_ls,y=f7_ls,yerr=df7_ls,fmt='o',label='N=7')
        # plt.errorbar(x=p8_ls,y=f8_ls,yerr=df8_ls,fmt='o',label='N=8')
        # plt.errorbar(x=p9_ls,y=f9_ls,yerr=df9_ls,fmt='o',label='N=9')
        p10_ls_full = np.array([1,2,3,4,5,6,7,8,9]) 
        f10_ls_full = reflect(f10_ls)
        df10_ls_full = reflect(df10_ls)
        plt.errorbar(x=p10_ls_full,y=f10_ls_full,yerr=df10_ls_full,ls='-',fmt='o',
                     label=r'SYM on $\mathbb{R}^3 \times \mathbb{S}^1$')
        plt.plot(p10_ls_full,sine_law2(p10_ls_full,10),marker='x',label='sine law')
        plt.plot(p10_ls_full,casimir2(p10_ls_full,10),marker='x',label='Casimir')
        plt.plot(p10_ls_full,np.sqrt(casimir2(p10_ls_full,10)),marker='x',label='sqrt Casimir')
        plt.xlabel('k')
        plt.ylabel('f')
        plt.title('f vs k (N=10)')
        plt.legend(bbox_to_anchor=(1,1.05))
        plt.tight_layout()
        plt.savefig('f vs k (N=10).png',dpi=500)
        plt.show()
    elif N == 4:
        #plot f as a function of k with different N series
        plt.figure()
        p4_ls_full = np.array([1,2,3]) 
        f4_ls_full = reflect(f4_ls)
        df4_ls_full = reflect(df4_ls)
        plt.errorbar(x=p4_ls_full,y=f4_ls_full,yerr=df4_ls_full,ls='-',fmt='o',
                     label=r'SYM on $\mathbb{R}^3 \times \mathbb{S}^1$')
        plt.plot(p4_ls_full,sine_law2(p4_ls_full,4),marker='x',label='sine law')
        plt.plot(p4_ls_full,casimir2(p4_ls_full,4),marker='x',label='Casimir')
        plt.plot(p4_ls_full,np.sqrt(casimir2(p4_ls_full,4)),marker='x',label='sqrt Casimir')
        plt.xlabel('k')
        plt.ylabel('f')
        plt.title('f vs k (N=4)')
        plt.legend(bbox_to_anchor=(1,1.05))
        plt.tight_layout()
        plt.savefig('f vs k (N=4).png',dpi=500)
        plt.show()
    
if __name__ == "__main__":
    #create dictionaries as global variables (used by functions)
    tension_dict = get_tension_dict()
    f_dict, df_dict = get_f_df_dict()
    #combine tension and f into master dictionary and array
    master_dict, master_array = master_dict_array()
    #print(master_dict)
    #print(master_array)
    
    #create a list of N corresponding to N-ality and the Tensions and f lists
    N1_list, T1_list, dT1_list, f1_list, df1_list = get_T_dT_f_df_by_p(1)
    N2_list, T2_list, dT2_list, f2_list, df2_list = get_T_dT_f_df_by_p(2)
    N3_list, T3_list, dT3_list, f3_list, df3_list = get_T_dT_f_df_by_p(3)
    N4_list, T4_list, dT4_list, f4_list, df4_list = get_T_dT_f_df_by_p(4)
    N5_list, T5_list, dT5_list, f5_list, df5_list = get_T_dT_f_df_by_p(5)

    # plot_tensions()
    plot_f()
    # plot_f_compare_sine()
    # plot_f_compare_sine_side_by_side()
    #plot_f_compare_casimir()
    #plot_f_compare_sqrt_casimir()
    # plot_f_compare_everything()

    #create a list of p that can exists in an N and the tensions and f lists
    p2_ls, T2_ls, dT2_ls, f2_ls, df2_ls = get_T_dT_f_df_by_N(2)
    p3_ls, T3_ls, dT3_ls, f3_ls, df3_ls = get_T_dT_f_df_by_N(3)
    p4_ls, T4_ls, dT4_ls, f4_ls, df4_ls = get_T_dT_f_df_by_N(4)
    p5_ls, T5_ls, dT5_ls, f5_ls, df5_ls = get_T_dT_f_df_by_N(5)
    p6_ls, T6_ls, dT6_ls, f6_ls, df6_ls = get_T_dT_f_df_by_N(6)
    p7_ls, T7_ls, dT7_ls, f7_ls, df7_ls = get_T_dT_f_df_by_N(7)
    p8_ls, T8_ls, dT8_ls, f8_ls, df8_ls = get_T_dT_f_df_by_N(8)
    p9_ls, T9_ls, dT9_ls, f9_ls, df9_ls = get_T_dT_f_df_by_N(9)
    p10_ls, T10_ls, dT10_ls, f10_ls, df10_ls = get_T_dT_f_df_by_N(10)
    
    plot_f_vs_p(N=10)
    plot_f_vs_p(N=4)

