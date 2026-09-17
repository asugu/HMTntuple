# HMTntuple

Analysis nTuple for HMT trigger studies. This repository can do the following:

 - Run HMT emulator
   - `CSCShowerAnalyzer` writes emulator-check histograms
 - Run CA4 clustering on HMT hits
 - Save emulated/L1 LCT hits
 - Run standard muon reco [experimental]
 - `SimpleCSCshowerFilter` builds HMT tree to contain the above

## Setup

```
cmsrel CMSSW_13_2_5
cd CMSSW_13_2_5/src
git clone git@github.com:kakwok/HMTntuple.git
cd HMTntuple/CSCShowerAnalyzer
scram b -j 8
```

## Running the code

For local testing, do:
```
cmsRun runCSCShowerAnalyzer_cfg.py inputFiles=file:~/eos/HLT/Commissioning2022/877caf18-4328-415e-bb74-b40adf51f9b4.root runNumber=351407 maxEvents=10
```

On lxplus, one can submit condor jobs with:
```
./cmsCondorData.py runCSCShowerAnalyzer_cfg.py $cmssw /eos/cms/store/user/kakwok/HLT/Commissioning2023/run370926  -n 2 -q workday -p /afs/cern.ch/user/k/kakwok/work/private/proxy -i list_370926
./sub_total.jobb
```
More example commands in `./test/run.sh`

To submit jobs on CRAB:

Load crab3 library
```
source /cvmfs/cms.cern.ch/crab3/crab.sh
```
The following command is to check crab configuration before submitting. If no issue has been observed, please remove `--dryrun` option at the end.
```
python3 crab_multi_dataset.py -p [pset.py] -o [eos/path/to/save/outputs] -t [tag:str] -i [input text file] --send-external -s [job splitting] -n [unit per job] --work-area [crab directory] --dryrun
```
The user can also set the run number by `-r [run number]` option. Also, if the user is giving private input files list, `--private` option can handle the situation. The format of input files is shown in the example directory under test.

**Caveat** When the user chooses to use `--private` option, the job splitting must be set to **FileBased**!! Also, please check the crab **whitelist** where to run jobs.

To check the CRAB job status:
```
python3 crab_multi_dataset.py --work-area [crab directory] --status
```
The failed job will be automatically resubmitted. If the user doesn't want auto-resubmission, please add `--no-resubmit` in the command line.


## CSC rechit cluster associations

See [CSC rechit-to-offline-cluster IDs](CSCShowerAnalyzer/docs/rechit-cluster-ids/README.md)
for the branch contract, implementation changes, separately supplied dependency,
100-event validation evidence and compatibility limits.
