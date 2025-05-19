Description.
==========
This library contains implementation of the mathematical model and tools to fits experimental data with a mathematical model.
It also contains a simulation part to simulate the same process through montecarlo sampling.
It contains visualisation tools.

Implementation of the mathematical model are in src/fast_rep/mathmod

It also contains an implementation of the mathematical model on a fix grid. It can be called using the command fit_rfd


Install
=========
# CPU
Does not work with newer scipy version...
```bash
mamba create -c conda-forge --name fast_rep_cpu  python jax jaxlib optax plotly numpy scipy=1.11.2 click typer pandas
mamba activate fast_rep_cpu
git clone https://github.com/organic-chemistry/fast_rep.git
cd fast_rep
pip install -e .
cd ..
git clone -b different-guides  https://github.com/organic-chemistry/jax_advi.git
cd jax_advi
pip install -e .
```


### For cuda and fit_ori tool only (not recommended):
```bash
mamba env create -f environment.yml
pip install -e .
pip install --upgrade chex optax plotly
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```



Example of notebook to fit data
==============================
Simple fit:
[Fit rfd](notebooks/bayesian/fit-experimental-rfd.ipynb)

Bayesian fit on synthetic data:
[Bayesian fit](notebooks/bayesian/baeysian_ori.ipynb)


[Using output from command line (example)](notebooks/bayesian/Compare-MAP-Laplace-ADVI.ipynb)



Reproducing Experiments
==============================

## Reproducing one experiment on the synthetic data at the command line:

Create synthetic data by fitting chrI from yeast.
```bash
#first fit the experimental data to produce a synthetic rfd relevant to yeast forkspeed 2500 bp/min S-phase 20 minutes
mode=MAP
model="Weibull"

fit_rfd_ori data/from_nfs_smv11.bed synthetic/chr1_fit.bed 2500 20 --regions chrI --fit-mode $mode --model-type $model
#visualisation of the fit:
visu_bed --chromosome chrI --start 0 --end 249000 --blocks "synthetic/chr1_fit.bed:original_rfd,synthetic/chr1_fit.bed:theo_rfd" --output compare.html
#visu firefor compare.html
```

Bayesian fitting of the synthetic data.
```bash
# then run the analysis on the synthetic data
model="Weibull"
fit_time="True"
bayesian_fit --input synthetic/chr1_fit.bed --regions chr1 --mode $mode --noise 0.075 --model $model --fit-time $fit_time --output synthetic/$model_$fit_time_bayesian.bed
```


## Comparing MAP Laplace ADVI

see [compare-MAP-Laplace-ADVI.sh](compare-MAP-Laplace-ADVI.sh) for the full combination of models.
Here is an example:
```bash
# then run the analysis on the synthetic data
model="Weibull"
fit_time="True"
bayesian  data/from_nfs_smv11.bed comparison/comp_MAP-${model}_${fit_time}_bayesian.bed 2500 20 --fit-mode MAP --regions chrI --model-type $model --fit-time --smoothv 19
bayesian  data/from_nfs_smv11.bed comparison/comp_Laplace-${model}_${fit_time}_bayesian.bed 2500 20 --fit-mode Laplace --regions chrI --model-type $model --fit-time --smoothv 19
bayesian  data/from_nfs_smv11.bed comparison/comp_ADVI-${model}_${fit_time}_bayesian.bed 2500 20 --fit-mode ADVI --regions chrI --model-type $model --fit-time --smoothv 19

#see notebooks/bayesian/Compare-MAP-Laplace-ADVI.ipynb for analysis and comparison

```
see [Using output from command line (example)](notebooks/bayesian/Compare-MAP-Laplace-ADVI.ipynb) for analysis and comparison

## Full genome analysis

see [full-genome.sh](full-genome.sh) to run it chromosome by chromosome (And eventually in parallel).


```bash
# then run the analysis on the synthetic data
XLA_FLAGS='--xla_force_host_platform_device_count=4'
model="Weibull"

fit_time="True"
bayesian  data/from_nfs_smv11.bed full-genome/Laplace-${model}_${fit_time}_bayesian.bed 2500 20 --fit-mode Laplace --model-type $model --fit-time --smoothv 19  

fit_time="False"
bayesian  data/from_nfs_smv11.bed full-genome/Laplace-${model}_${fit_time}_bayesian.bed 2500 20 --fit-mode Laplace --model-type $model  --smoothv 19  

model="Exponential"
fit_time="True"
bayesian  data/from_nfs_smv11.bed full-genome/Laplace-${model}_${fit_time}_bayesian.bed 2500 20 --fit-mode Laplace --model-type $model --fit-time --smoothv 19  

fit_time="False"
bayesian  data/from_nfs_smv11.bed full-genome/Laplace-${model}_${fit_time}_bayesian.bed 2500 20 --fit-mode Laplace  --model-type $model  --smoothv 19  
#see notebooks/bayesian/Compare-MAP-Laplace-ADVI.ipynb for analysis and comparison

```

Usage for fixed grid
==============================
MP.rfd.for_JM.bedGraph  == bedgraph with rfd
MP.rfd_n.for_JM.bedGraph  == bedgraph with count not necessary

```sh
fit_rfd MP.rfd.for_JM.bedGraph test.bedGraph 1 600  MP.rfd_n.for_JM.bedGraph  --reg-loss 100000 --regions chr1:0-15000000 --tolerance 0.001 --floor-v 0.0000000001 --ar-sigma 1 --flat
```

```sh
visu_bed   --chromosome chr1   --start 0   --end 13000000   --blocks "test.bedGraph:original_rfd,test.bedGraph:theo_rfd" "test.bedGraph:theo_mrt" "test.bedGraph:lambdai" "test.bedGraph:weight"  "MP.rfd_n.for_JM.bedGraph:signal"  --output output.html --resolution 1
```


Simulate with exponontial model
```sh
python src/scripts/simulate.py test.bedGraph test_sim.bed 1. --n-sim 10000
```

Visualise and compare
```sh
visu_bed   --chromosome chr1   --start 0   --end 13000000   --blocks "test.bedGraph:original_rfd,test.bedGraph:theo_rfd,test_sim.bed:simu_rfd" "test.bedGraph:theo_mrt,test_sim.bed:simu_mrt" "test.bedGraph:lambdai" "test.bedGraph:weight"  "MP.rfd_n.for_JM.bedGraph:signal"  --output output.html --resolution 1
```sh