#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
export MYPYTHON=/afs/cern.ch/work/c/camontan/public/anaconda3

unset PYTHONHOME
unset PYTHONPATH
source $MYPYTHON/bin/activate
export PATH=$MYPYTHON/bin:$PATH

which python

# echo the argument received by the script
echo $1

python3 /afs/cern.ch/work/c/camontan/public/tangent_map/run_sim.py --config $1

eos cp *.h5 /eos/user/c/camontan/lhc_dynamic_data

rm *.h5