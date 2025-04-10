# general imports
import math
from GATraning_func import fitness
from SineFunc_Traning import MSE_fit, Sine_funcA, fitnessWave, QRMS
import numpy as np # type: ignore



# extract from fake data
from Fake_data import FakeD, Data_Pol_deg,trainig_data_fake

#extract data from PW
#from exportPW import Mods_Pw, trainig_modes

#extract data from ZW
from exportZW import trainig_modes

# Fake data generator
Artif_data = FakeD()
RescProb=0.3
Full_data=Data_Pol_deg(Artif_data) #Full data without reescaling
sF_pts=trainig_data_fake(Artif_data,Full_data,RescProb)


#random angles of the circuit 
def Rand_param(no_parameters):
    return np.random.uniform(0, 2*np.pi, no_parameters)




#This is the function that one used to find the optimal angle 
# for mimimize the error in the predictions

def opt_ang(Pred_wave,m,l):
    x = np.arange(0,2*np.pi,0.0005)
    y=fitnessWave(x,Pred_wave,m,l)
    x_opt=x[np.argmin(y)]  
    return x_opt



class Sine_Func_Pred:
 """Sine Function predictied
    attributes: Amplitude, angle and constant
    """
sF_Pred_Vec=Sine_Func_Pred()


#Main functions of Sine wave approach
#Samp gives the number of samples used - samp=0 goes to ideal prob.
def Prel_main(Parametrized_circ,upd_ang,samp,period,m,l):
    Data=trainig_modes(m,l) #data ZW
    size_data=len(Data.OutTrset)
    A=np.zeros(size_data)
    phi=np.zeros(size_data)
    c=np.zeros(size_data)
    for j in range(0,size_data):
        sF=Sine_funcA(Parametrized_circ,upd_ang,j,samp,period,m,l)
        A[j]=sF.amplitude
        phi[j]=sF.phase
        c[j]=sF.const
    sF_Pred_Vec.amplitude=A
    sF_Pred_Vec.phase=phi
    sF_Pred_Vec.const=c
    return sF_Pred_Vec




#inputs:
#angles: select the angles in the circuit that should be optimized
# m is the mode used and l is the choice of training set l=0; 5 points; l=1 7 points; l=2 9 points
#Samp: gives the number of samples used - samp=0 goes to ideal prob.
#rounds: the maximum of rounds regardelss if the theshold was achieved
def main_SF(threshold,circ_parem,angles,rounds,m,l,samp):
    #Data=trainig_modes(m,l) #data PW
    new_param=np.zeros(len(circ_parem))+circ_parem
    j=1
    tr=0
    should_we_stop = False
    while not(should_we_stop or j>rounds):
        for k in angles:
            Pred_sF=Prel_main(new_param,k,samp,1,m,l)
            new_param[k]= opt_ang(Pred_sF,m,l)
        j=j+1
        fit = MSE_fit(new_param,0.5,m,l)
        should_we_stop = np.min(fit) <= threshold
    if np.min(fit) <= threshold:
        tr=1
        print('Threshold achieved SF')
    return [new_param,j,tr,fit]
#Outcomes:
#new_param: New circuit
#j: number of rounds
#tr: tells if the threshold was achieved



#from SineFunc_Traning import MSE_fit
# from SineFunc_Traning import QRMS
# from Main_sFunc_traning import main_SF
# from Main_sFunc_traning import Rand_param 
# from Main_Ga_traning import Verificator
#Circ2=main(Circ,[0,1,2,3,4,5,6,7],80,0,0)
#QRMS(Circ2,0.5,0,0)

#thesh_SF: gives the expected thresolds for the traing points
#Param_circ-it is a parametrized circuit that has to be used
def RunSf(Sf_rep,thesh_SF,Param_circ,m,l,samp):
    RMS=[]
    t_sf=[]
    j=1
    cost=0
    while j<50:
        #tr tells if the threshold was achieved
        #t_sf tells the number of rounds
        [Circ,rounds,tr_sf]=main_SF(thesh_SF,Param_circ,[0,1,2,3,4,5,6,7],Sf_rep,m,l,samp)
        rms=QRMS(Circ,0.5,m,l)
        cost=cost+1
        if tr_sf==1:
            RMS.append(rms)
            t_sf.append(rounds)
            j=j+1
    return [RMS,t_sf,cost]
