#!/bin/bash

python all_smart.py --idx 0 &
python all_smart.py --idx 1 &
python all_smart.py --idx 2 &
python all_smart.py --idx 3 &
python all_smart.py --idx 4 &
python all_smart.py --idx 5 &
python all_smart.py --idx 6 &
python all_smart.py --idx 7 &

wait
echo "Done"