#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_102cuda/x86_64-centos7-gcc8-opt/setup.sh
source /afs/cern.ch/work/c/camontan/public/tangent_map/myenv/bin/activate
export EOS_MGM_URL=root://eosuser.cern.ch

which python

# echo the argument received by the script
echo $1

python3 /afs/cern.ch/work/c/camontan/public/tangent_map/run_sim.py --config $1

eos cp *.h5 /eos/user/c/camontan/lhc_dynamic_data

rm *.h5