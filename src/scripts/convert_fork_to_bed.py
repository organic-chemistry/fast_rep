from fast_rep.rfd_tools  import load_RFD
from fast_rep.read_data import write_custom_bedgraph_pandas
import numpy as np

data = "/home/jarbona/alignExp/notebook/invert_conf/data/NFS/NFS_BTmulti_"
smv=11
output_file = f"data/from_nfs_smv{smv}.bed"


RFD, mean_RFD, Cov_RFD, std_RFD,positif,negatif = load_RFD(root=data, minimum_obs=1)
_,smth_RFD,_,_,_,_ =  load_RFD(root=data,minimum_obs=1,smv=11)

final =  {}

resolution = 100
for key in RFD:
    final[key] = {"signals":{"RFD":RFD[key],
                             "smth_RFD":smth_RFD[key],
                             "std_RFD":std_RFD[key]},
                  "chrom":key,
                  "start":np.arange(len(RFD[key]))*resolution,
                  "end":np.arange(len(RFD[key]))*resolution + resolution}
    

write_custom_bedgraph_pandas(output_file,final)

