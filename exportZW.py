import numpy as np
import csv

def rows_Zw(x):
    Vec=[]
    try: 
        fin=open('Zw_data/trainLabelsv2_36.csv')
    except: 
        print('Something went wrong.')
    with fin as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each colunm is a modo
            Vec.append(row)
        Modo_vec=Vec[x+1]
    return Modo_vec

def Mods_Zw(m):
    Mod=[]
    for row in range(36):
        Mod.append(rows_Zw(row)[m+1])
    return Mod

class InOutModes_Resc:
    """Represents the all the input and output data used for the tranining.
    attributes:  full data rescaled Input: InFdata, Output: OutFdata; Traning_Set Input: InTrset Output: OutTrset; Rescaling used resc
    """
Modes_Resc=InOutModes_Resc() 

def traning_labels(l):
    Traning=[[1,3,6,9,12,15,18,21,24,27,30,33],[0,1,3,5,6,9,12,14,15,18,21,24,27,29,30,31,33,35],[0,1,3,4,5,6,9,11,12,14,15,17,18,19,21,22,24,27,28,29,30,31,33,35]]
    return Traning[l]



def trainig_modes(m,l,Total_points=36,resc_Prob=0.5):
    Train_Labels=traning_labels(l)
    Full_inputdata= [number for number in Mods_Zw(-1)]
    Full_outdata=Mods_Zw(m)
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