# general imports
import numpy as np # type: ignore
import random

# extract from fake data
from Fake_data import trainig_data_fake
from Fake_data import Data_Pol_deg


# extract from circuits op
from Basic_Gates import measurement
from Basic_Gates import simulate

#Choose the circuit
#from Dirk_Circuit import the_circuit 
from DirkPower_Circuit import the_circuit 


# extract functions from the Ga_traning
from GATraning_func import init_pop
from GATraning_func import fitness
from GATraning_func import update_pop

#extract data from PW
#from exportPW import Mods_Pw, trainig_modes

#extract data from ZW
from exportZW import Mods_Zw, trainig_modes



#PW data
#Ga_pts=trainig_modes(16,l) #Rescaling used
#Full_dataOut=Mods_Pw(15)
   



#Resc_prob for fake data 0.4
#Resc_prob for ZW and PW mode 0.5
def QRMS(circ_parem,m,l,resc_Prob=0.5):
    Data=trainig_modes(m,l)
    #Out=Mods_Pw(m)#QRMS for Pw data
    Out=Mods_Zw(m)#QRMS for Zw data
    RMS=0
    for j in range(len(Data.OutFdata)):
        Pr = measurement(simulate(the_circuit(Data.InFdata[j],circ_parem),3),2,3)
        target_prob = Out[j]
        RMS += (10**Data.resc*(Pr - resc_Prob) - target_prob)**2
    return [np.sqrt(RMS/len(Data.OutFdata))]

#resc for GA PW 0.5
#resc for SF PW 0.3
def Verificator(circ_parem,m,l,resc_Prob=0.3):
    Data=trainig_modes(m,l)
    Out=Mods_Pw(m)
    Pred_out=[]
    for j in range(len(Data.InFdata)):
        Pr = measurement(simulate(the_circuit(Data.InFdata[j],circ_parem),3),2,3)
        Pred_out.append(10**Data.resc*(Pr - resc_Prob))
    return [Pred_out,Out]



class Fake:
    """Represents the Fake data.
    attributes: Coef_poly (Coefficient of the polynomial), domain, points (total points used for the total data),traning_Points
    """
Fake_data=Fake()
Fake_data.Coef_poly=[1,3,6,2]
Fake_data.domain=[-2,3]
Fake_data.points=86
Fake_data.traning_Points=sorted(random.sample(range(86), 42))





#Full_data=Data_Pol_deg(Fake_data)
#[input_data,Resc_inp,output_data,Resc_out]=trainig_data_fake(Fake_data,Full_data)
# m is the mode used and l is the choice of training set l=0; 5 points; l=1 7 points; l=2 9 points
def main_GA(threshold,T,sigma,pm,pc,m,l,samp):
    Data=trainig_modes(m,l)
    pop=init_pop(23,80)
    i=1
    tr=0
    should_we_stop = False
    while not(should_we_stop or i>T):
        fit = fitness(pop,Data.InTrset,Data.OutTrset,2,3,samp)
        pop_save=pop
        should_we_stop = np.min(fit) <= threshold
        pop=update_pop(pop,fit,sigma,pm,pc)
        i=i+1
    if np.min(fit) <= threshold:
        tr=1
        print('Threshold achieved GA')
    return [pop_save,fit,i,tr]



def RunGA(Trs,T_ga,m,l,samp):
    rms=[]
    t_ga=[]
    j=1
    cost=0
    while j<21:
        [Pop,fit,t_f,tr_ga]=main_GA(Trs,T_ga,0.665,0.38,0.08,m,l,samp)
        cost=cost+1
        #tr_ga certifies that we only consider the cases that succeed the threshold values
        if tr_ga==1:
            index_min = min(range(len(fit)), key=fit.__getitem__)
            r=QRMS(Pop[index_min],m,l)
            rms.append(r)
            t_ga.append(t_f)
            j=j+1
    return [rms,t_ga,cost]
#1) (0,0) hyperparem sigma=0.435,pm=0.48,pc=0.05
#0.0000132 - 5k samples
#0.00005 - 2k samples
#2) (1,0) hyperparem sigma=0.278,pm=0.55,pc=0.08
#0.00008 - 5k samples
#0.00009 - 2k samples