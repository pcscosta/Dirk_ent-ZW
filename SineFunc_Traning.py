# general imports
import math
import numpy as np # type: ignore
import random
import matplotlib.pyplot as plt # type: ignore


# extract from circuits op
from Basic_Gates import measurement, simulate


#Choose the circuit
#from Dirk_Circuit import the_circuit #This was used for PW
from DirkPower_Circuit import the_circuit 

#Choose data
from Fake_data import Data_Pol_deg, FakeD, trainig_data_fake

#extract data from PW
#from exportPW import Mods_Pw, trainig_modes

#extract data from ZW
from exportZW import Mods_Zw, trainig_modes



# Fake data generator
Artif_data = FakeD()
RescProb=0.3
Full_data=Data_Pol_deg(Artif_data) #Full data without reescaling
sF_pts=trainig_data_fake(Artif_data,Full_data,RescProb)



#This function is used to determine the sine wave considering the three points

class Sine_Func:
 """Sine Function predictied
    attributes: Amplitude, angle and constant
    """
sF_Parm=Sine_Func()


# This function choose a specific parameter of the quantum circuit (upd_ang), for a given data point (dx_input) and returns the wave fittted wave
def Sine_funcA(Parametrized_circ,upd_ang,idx_input,samp,period,m,l,qt=3,q=2):
    #Data=sF_pts Fake data. This requeries another use of a class
    Data=trainig_modes(m,l)#data PW
    new_param=np.zeros(len(Parametrized_circ))+Parametrized_circ
    Y=np.zeros(3)
    X=[0.5,0.5+math.pi,0.5+math.pi/2]
    j=0;
    for i in X:
        new_param[upd_ang]=i
        if samp != 0:
            p = measurement(simulate(the_circuit(Data.InTrset[idx_input],new_param),qt),q,qt)
            Y[j]=sum(np.random.binomial(1, p, samp))/samp
        else:
            Y[j]=measurement(simulate(the_circuit(Data.InTrset[idx_input],new_param),qt),q,qt)
        j=j+1
    if abs(Y[0]-Y[1])<=10**-5 or abs(Y[1]-Y[2])<=10**-5:
        A=1
        phi=1
        c=1
    else:
        c= (Y[0]+Y[1])/2;
        phi=np.arctan((Y[0]-c)/(Y[2]-c))-X[0]/period;
        A=(Y[1]-c)/math.sin(X[1]/period+phi);
        sF_Parm.amplitude=A
        sF_Parm.phase=phi
        sF_Parm.const=c
    return sF_Parm


#This function is used to generate the plots of the sine function behavior of each angle parameter

def Pr(circ_parem,ang):
    new_param=np.zeros(len(circ_parem))+circ_parem
    theta= np.linspace(0, 2*math.pi, 100);
    vec=np.zeros(theta.shape)
    j=0;
    for i in theta:
        new_param[ang]=i
        vec[j]=MSE_fit(new_param,0)
        j=j+1
    fig = plt.figure(figsize=(8, 6))
    plt.plot(theta,vec,marker='D')
    plt.ylabel('MSE')
    plt.xlabel('# Angle paremeter from Dirk')
    plt.show()
    return fig


# Quality functions



#This function considers the MSE only of the trainig points
def MSE_fit(circ_param,resc_Prob,m,l,q=2,qt=3):
    #Data=sF_pts Fake data. This requeries another class use
    Data=trainig_modes(m,l)#Either a ZW or a PW data 
    MSE=0
    for idx_input in range(len(Data.OutTrset)):
        Pr = measurement(simulate(the_circuit(Data.InTrset[idx_input],circ_param),qt),q,qt)
        target_prob = resc_Prob + Data.OutTrset[idx_input]
        MSE += (Pr - target_prob)**2
    return np.abs(MSE/len(Data.OutTrset))





#This function consider the sine function estimated so it can be used to estimated what is 
# the x values (angles) that will minimize the cost function for the traning points only
def fitnessWave(x,Pred_wave,m,l,resc_Prob=0.5):
    #Data=sF_pts Fake data. This requeries another class use
    Data=trainig_modes(m,l)#either a Zw or a PW data
    output = 0
    for j in range(len(Data.OutTrset)):
        Pr = Pred_wave.amplitude[j]*np.sin(x+Pred_wave.phase[j])+Pred_wave.const[j]
        target_prob = resc_Prob + Data.OutTrset[j]
        output += (Pr - target_prob)**2
    return np.sqrt(output/len(Data.OutTrset))



#This function considers the QRMS of the entilely set
def QRMS(circ_parem,resc_Prob,m,l):
    #Data=sF_pts Fake data. This requeries another class use
    Data=trainig_modes(m,l)#either zw or Pw data
    Out=Mods_Zw(m)# output modes without rescaling
    RMS=0
    for j in range(0,len(Data.OutFdata)):
        Pr = measurement(simulate(the_circuit(Data.InFdata[j],circ_parem),3),2,3)
        target_prob =Out[j]
        RMS += (10**Data.resc*(Pr - resc_Prob) - target_prob)**2
    return np.sqrt(RMS/len(Data.OutFdata))


# I am not making use of it
def Verificator(circ_parem,m,l,resc_Prob=0.5):
    Data=trainig_modes(m,l)
    #Out=Mods_Pw(m)#QRMS for Pw data
    Out=Mods_Zw(m)
    Pred_out=[]
    for j in range(len(Data.InFdata)):
        Pr = measurement(simulate(the_circuit(Data.InFdata[j],circ_parem),3),2,3)
        Pred_out.append(10**Data.resc*(Pr - resc_Prob))
    return [Pred_out,Out]






#Values.append(Hybrid_Syst(0.0001,6,0.475,0.48,0.08,60,0,0))
