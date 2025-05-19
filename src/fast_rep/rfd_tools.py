import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from fast_rep.math_mod.compute_rfd import compute_derivatives

def convert_RFD_delta_MRT(pos_to_compute,data,speed,resolution,delta=15,measurement_type="deltaMRT"):
    
    MRT = np.cumsum(data)[::delta]   #
    if measurement_type == "deltaMRT":
        MRT /= (speed/resolution)
    else:
        MRT /= delta
    #print(MRT[:5])
    Deltas =  compute_derivatives(MRT,method="central",v=1,shift=delta)
    return pos_to_compute[::delta],Deltas

def smooth(ser, sc):
    return np.array(pd.Series(ser).rolling(sc, min_periods=1, center=True).mean())

def load_RFD(root="data/NFS_WTx10/BT_P2_WTx10_refBT1mono_", minimum_obs=20,smv=1):
    # Load experimental RFD
    RFD_r = pd.read_csv(f"{root}rfd_nt.csv")
    Cov_r = pd.read_csv(f"{root}covT_nt.csv")

    RFD = {}
    for ch in RFD_r['chrom'].unique():
        chrom_str = str(ch)
        df_chrom = RFD_r[RFD_r['chrom'] == ch]
        signal = df_chrom['signal'].fillna(0).astype(float).values
        RFD[chrom_str] = signal

    Cov_RFD = {}
    mean_RFD = {}
    std_RFD = {}
    Pos = {}
    Neg = {}

    for ch in Cov_r['chrom'].unique():
        chrom_str = str(ch)
        df_cov = Cov_r[Cov_r['chrom'] == ch]
        cov_signal = df_cov['signal'].astype(float).values
        Cov_RFD[chrom_str] = smooth(cov_signal,smv)

        # Get corresponding RFD data
        rfd_signal = RFD[chrom_str]
        n0 = cov_signal
        p = (1 + rfd_signal) / 2
        np_val = minimum_obs

        # Calculate positif and negatif with prior
        positif = smooth(np.round(n0 * p + np_val).astype(int),smv)
        negatif = smooth(np.round(n0 * (1 - p) + np_val).astype(int),smv)

        Pos[chrom_str] = positif
        Neg[chrom_str] = negatif

        # Calculate mean RFD
        mean_rfd = 2 * positif / (positif + negatif) - 1
        mean_RFD[chrom_str] = mean_rfd

        # Calculate standard deviation
        proba = (1 + mean_rfd) / 2
        std_rfd = np.sqrt(proba * (1 - proba) / (n0 + 2 * np_val))
        std_RFD[chrom_str] = smooth(std_rfd,smv)
        if smv != 1:
            RFD[ch] = smooth(RFD[ch],smv)
            RFD[ch] = smooth(RFD[ch],smv)


    return RFD, mean_RFD, Cov_RFD, std_RFD,Pos,Neg


def find_ori_position(data={},smoothv=11,min_dist_ori=30,min_rfd_increase_by_kb=0.1):
    if smoothv % 2 == 0:
        raise "smoothv must be impair"

    rfd = data["rfd"]
    delta= np.concatenate([[0],rfd[2:]-rfd[:-2],[0]])/2
    delta = smooth(delta,smoothv)
    #delta -=  smooth(delta,2*smoothv)

    pos = data["positions"]
    resolution =  pos[1:]-pos[:-1]
    assert(np.all(resolution==resolution[0]))
    resolution = resolution[0]
    delta *= 1000/resolution  # to have increase in rfd/kb
    #p2=plot(delta)
    pks, vals = find_peaks(delta,distance=min_dist_ori/resolution,prominence=min_rfd_increase_by_kb)
    #pks,delta
    return data["positions"][pks],delta,vals


def extract_most_potent_ori_around_expected_value(xis,amplitude,expected_n_ori,
                                                  max_factor_expected=2.5,
                                                  min_factor_expected=0.7):

    
    maxi = min(int(max_factor_expected * expected_n_ori),len(xis))
    mini = max(1,int(min_factor_expected * expected_n_ori))
    print(maxi,mini)
    Ori_pos = []
    for n_ori in list(range(mini,maxi))[::-1]:
        Ori_pos.append(xis[np.argsort(amplitude)[::-1][:n_ori]])
    return Ori_pos