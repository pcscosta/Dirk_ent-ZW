

from Main_Ga_traning import main_GA
from SineFunc_Traning import QRMS
from Main_sFunc_traning import main_SF


def Hybrid_Syst(thesh_GA,T_ga,var,pm,pc,thesh_SF,SF_rep,m,l,samp_ga,samp_sf):
    [Pop,fit,t_ga,tr_ga]=main_GA(thesh_GA,T_ga,var,pm,pc,m,l,samp_ga)
    index_min = min(range(len(fit)), key=fit.__getitem__)
    [Circ,t_sf,tr_sf,fit]=main_SF(thesh_SF,Pop[index_min],[0,1,2,3,4,5,6,7,9,10,11,12,13,14,16,17,18,19,20],SF_rep,m,l,samp_sf)
    rms=QRMS(Circ,0.5,m,l)
    return [rms,t_ga-1,t_sf-1,tr_ga,tr_sf,fit]


def RunHybrid(T_ga,Sf_rep,m,l,samp_ga,samp_sf):
    rms=[]
    t_ga=[]
    t_sf=[]
    j=1
    cost=0
    while j<11:
        [r,t1,t2,tr_ga,tr_sf,fit]=Hybrid_Syst(0.0031,T_ga,0.475,0.38,0.05,0.0025,Sf_rep,m,l,samp_ga,samp_sf)
        cost=cost+1
        if tr_sf==1:
            rms.append(r)
            t_ga.append(t1)
            t_sf.append(t2)
            j=j+1
    return [rms,t_ga,t_sf,cost]

#[rms,t_ga,t_sf]=Hybrid_Syst(0.0003,10,0.475,0.48,0.08,0.000001,100,0,0)
# (0,0) Thrshold ga=0.0005;  Threshold sf= 0.000002
# (1,0) Thrshold ga=0.0035;  Threshold sf= 0.00006

#Limited sample
# (0,0) [r,t1,t2,tr_ga,tr_sf]=Hybrid_Syst(0.01,10,0.475,0.48,0.08,0.00001,30,0,0,1000,600)
# (1,0) [rms,t_ga,t_sf,tr_ga,tr_sf]=Hybrid_Syst(0.08,10,0.475,0.48,0.08,0.0003,20,1,0,1000,600)
# (2,0) [rms,t_ga,t_sf,tr_ga,tr_sf]=Hybrid_Syst(0.05,10,0.475,0.48,0.08,0.008,20,2,0,1000,600)

#Limited sample ZW
# (0,0) Hybrid_Syst(0.002,4,0.475,0.38,0.05,0.0002,8,1,0,200,650)
# (1,0) Hybrid_Syst(0.0012,4,0.475,0.38,0.05,0.0003,8,1,0,200,650)
# (2,0) Hybrid_Syst(0.015,4,0.475,0.38,0.05,0.0135,8,1,0,200,650)
# (3,0) Hybrid_Syst(0.019,4,0.475,0.38,0.05,0.018,8,1,0,200,650)
# (4,0) Hybrid_Syst(0.0058,4,0.475,0.38,0.05,0.0040,8,1,0,200,650)

#[0,1,2,3,4,5,7,8,9,10,11,12,13,14,16]