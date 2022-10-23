#!/bin/bash

python all.py --idx 0 &
python all.py --idx 1 &
python all.py --idx 2 &
python all.py --idx 3 &
python all.py --idx 4 &
python all.py --idx 5 &
python all.py --idx 6 &
python all.py --idx 7 &

wait
echo "Done"