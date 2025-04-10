import numpy as np
import csv




def Mods_Pw(x):
    Vec=[]
    try: 
        fin=open('PW_data/ProjectionCoefficients.csv')
    except: 
        print('Something went wrong.')
    with fin as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each colunm is a modo
            Vec.append(row)
        Modo_vec=Vec[x+1]
        del Modo_vec[0]
    return Modo_vec

class InOutModes_Resc:
    """Represents the all the input and output data used for the tranining.
    attributes:  full data rescaled Input: InFdata, Output: OutFdata; Traning_Set Input: InTrset Output: OutTrset; Rescaling used resc
    """
Modes_Resc=InOutModes_Resc() 


def traning_labels(l):
    Traning=[[1,6,10,13,20],[1,4,6,10,13,17,20],[1,4,6,8,10,13,15,17,20]]
    return Traning[l]


# The resc_Prob 0.5 is used for the PW with the 3qubits gate
#l is the label of traning points to be used and m is the modes used
def trainig_modes(m,l,Total_points=21,resc_Prob=0.5):
    Train_Labels=traning_labels(l)
    Full_inputdata= [10*number for number in Mods_Pw(-1)]
    Full_outdata=Mods_Pw(m)
    Full_outdata_resc=Full_outdata
    OutTraning=[]
    InTraning=[]
    resc=-1
    while max(np.absolute(Full_outdata_resc)) > resc_Prob:
        TempData=[]
        resc+=1
        for j in range(Total_points):
             TempData.append((10**-resc)*Full_outdata[j])
        Full_outdata_resc=TempData
    if resc <0:
        resc=0
    for train_points in Train_Labels:
        OutTraning.append(Full_outdata_resc[train_points])
        InTraning.append(Full_inputdata[train_points])
    Modes_Resc.OutFdata=Full_outdata_resc
    Modes_Resc.OutTrset=OutTraning
    Modes_Resc.InFdata=Full_inputdata
    Modes_Resc.InTrset=InTraning
    Modes_Resc.resc=resc
    return  Modes_Resc