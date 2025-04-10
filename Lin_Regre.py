from exportPW import Mods_Pw
import math
import scipy.integrate as integrate
import numpy as np

PWtraning_Set5=[1,6,10,13,20]
PWtraning_Set7=[1,4,6,10,13,17,20]
PWtraning_Set9=[1,4,6,8,10,13,15,17,20]

class InOut_Data:
    """Represents the all the input and output data used for the tranining.
    attributes:  full data rescaled Input: InFdata, Output: OutFdata; Traning_Set Input: InTrset Output: OutTrset; Rescaling used resc
    """
Modes_Resc=InOut_Data() 

Modes_Resc.Inp=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2]
Modes_Resc.TrSet=PWtraning_Set5





def Basis_Penalty(n,Io,If):
    P= np.array([])
    h = 1e-5
    for k in range(n):
        result = integrate.quad(lambda x: ((x+h)**k-2*(x)**k+(x-h)**k)/(h**2), Io, If)
        P=np.append(P,result[0])
    return P


def Basis(n,x):
    P= np.array([])
    for k in range(n):
        P=np.append(P,x**k)
    return P

def Trig_Basis(n,x):
    P= np.array([])
    P=np.append(P,1)
    for k in range(1,n):
        P=np.append(P,math.cos(x)**k)
        P=np.append(P,math.sin(x)**k)
    return P

def Trig_Basis2(n,x):
    P= np.array([])
    P=np.append(P,1)
    for k in range(1,n):
        P=np.append(P,math.cos(k*x))
        P=np.append(P,math.sin(k*x))
    return P

def Coef_LinReg(ord,md,lam,Data=Modes_Resc):
    Out=Mods_Pw(md)
    X=np.zeros(ord)
    X_pen=np.zeros(ord)
    TrInp=[]
    for k in Data.TrSet:
        N_row=Basis(ord,Data.Inp[k])
        X = np.vstack([X,N_row])
        TrInp.append(Out[k])
    X = np.delete(X, (0), axis=0)
    #mn=float(min(Out))
    #mx=float(max(Out))
    X_pen=Basis_Penalty(ord,0,0.2)
    Omega=np.transpose(X_pen).dot(X_pen)
    dim=np.shape(X)
    return [Omega,np.linalg.inv(np.transpose(X).dot(X)- lam*Omega*np.identity(dim[1])).dot(np.transpose(X)).dot(TrInp)]


def Rms_LinReg(Coef,n,md,Data=Modes_Resc):
    Out=Mods_Pw(md)
    rms=0
    j=0
    for k in Data.Inp:
        rms+= (Coef.dot(Basis(n,k))-Out[j])**2
        j+=1
    return np.sqrt(rms/len(Data.Inp))

def Rms_LinReg2(Coef,n,md,Data=Modes_Resc):
    Out=Mods_Pw(md)
    rms=0
    j=0
    for k in Data.Inp:
        rms+= (Coef.dot(Basis(n,k))-Out[j])**2
        j+=1
    return np.sqrt(rms/len(Data.Inp))


