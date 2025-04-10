import numpy as np

#Input parameters for Data_Pol_deg:
    #1 Coef_Pol is a list where the values are the coeffients of the polynomial of degree len(Coef_Pol)-1
    #2 Points gives the number of points will be used for the fake data
    #3 Domain gives the domain used to generate the fake data


class PointsFake:
    """Represents the all the input and output data used for the tranining.
    attributes: Coef_poly (Coefficient of the polynomial), domain, points (total points used for the total data),traning_Points
    """
in_out_Points=PointsFake()  



def Data_Pol_deg(Fake_data):
    T_Points = Fake_data.points -1 
    InterV= (Fake_data.domain[1]-Fake_data.domain[0])/T_Points

    All_Points_out=[]
    All_Points_in=[]
    Entry = Fake_data.domain[0]
    for k in range(Fake_data.points):
        All_Points_in.append(Entry)
        Val=0
        power=0
        for j in Fake_data.Coef_poly:
            Val += j*Entry**power
            power+=1
        Entry +=  InterV
        All_Points_out.append(Val)
    in_out_Points.input=All_Points_in
    in_out_Points.output=All_Points_out 
    in_out_Points.input=All_Points_in
    in_out_Points.output=All_Points_out

    return in_out_Points




def Resc_function(Full_data,Resc_Targ):
    Final_data= Full_data
    resc=-1
    while max(np.absolute(Final_data)) > Resc_Targ:
        train_data = []
        resc+=1
        for train_points in Full_data:
            train_data.append((10**-resc)*train_points)
        Final_data=train_data
    if resc <0:
        resc=0
    return  [train_data,resc]


class sineF_TraningPoints:
    """Represents the Fake data used in the sine function traning
    attributes: input (domain-D of the function) in_Resc (reescaled value used in the domain which is used in the circuit to guarantee that the abs(max{D})<=1)
    output_data (image of polynomial used) 
    """
sF_pts=sineF_TraningPoints()

# Input parameter of trainig_data_fake:
    #1 Train_Labels is a list selecting the position of the full data which are going to be used for the training
    #2 Full_data is a list with the list of full imput data and the full output data
    #3 Resc_Prob is the rescaling to make the negative values for the output data become positive in case there are negative values in the domeian, otherwise it should be equal to zero
    #4 Resc_Inp is the rescaling to make the values in the inp_data used in the circuit smalles than abs(1) 

def trainig_data_fake(Fake_data,in_out_Points,Resc_Prob=0.4,Resc_Inp = 1):
    Train_data_Inp = []
    Train_data_Out = []
    [Full_data_in,Resc_inp] = Resc_function(in_out_Points.input,Resc_Inp)
    [Full_data_out,Resc_out] = Resc_function(in_out_Points.output,Resc_Prob)
    for j in Fake_data.traning_Points:
        Train_data_Inp.append(Full_data_in[j])
        Train_data_Out.append(Full_data_out[j])

    sF_pts.input=Train_data_Inp
    sF_pts.in_Resc=Resc_inp
    sF_pts.output_data=Train_data_Out
    sF_pts.out_Resc=Resc_out
    return sF_pts
    




class Fake:
    """Represents the Fake data.
    attributes: Coef_poly (Coefficient of the polynomial), domain, points (total points used for the total data),traning_Points
    """
Fake_data=Fake()

#Fake data inputs orgazization in a class

def FakeD(Coef_Poly=[1,3,6,2],Dom=[-2,3],Tot_P=86,Tran_Set=[0,2,5,6,8,9,10,12,14,15,16,18,19,20,21,23,24,26,28,30,31,33,35,38,40,42,43,45,47,48,50,52,55,58,60,63,64,65,68,69,70,72,74,78,80,82,83,85]):
    Fake_data.Coef_poly=Coef_Poly
    Fake_data.domain=Dom
    Fake_data.points=Tot_P
    Fake_data.traning_Points=Tran_Set
    return Fake_data
#sorted(random.sample(range(86), 68))



    return  [Train_data_Inp,Resc_inp,Train_data_Out,Resc_out]
