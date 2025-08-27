#!/bin/sh
#
# The code in this repository is accompanying the publication "Frequency
# Assignment for Guaranteed QoS in Two-Ray Models with Limited Location
# Information" (K.-L. Besser, E. Jorswieck, J. Coon, H. V. Poor. May 2025,
# WiOpt 2025. DOI:10.23919/WiOpt66569.2025.11123355)
#
# License of the code: MIT


FREQ0="26.5e9"
HTX=10
HRX="1.5"

NUM_RUNS=250

echo "Figure 2: Distances of destructive interference"
python example_local_minima.py -v -t "$HTX" -r "$HRX" -dmin 35 -dmax 85 -f 8e9 -df 800e6 -n 3 --plot --export

echo "Table I: Worst-case receive power"
python assign_frequencies.py -f "$FREQ0" -df 10e6 -n 300 -dmin 45 -dmax 50 -t "$HTX" -r "$HRX" -m 16 --num_runs="$NUM_RUNS" -v --export
python assign_frequencies.py -f "$FREQ0" -df 10e6 -n 300 -dmin 90 -dmax 100 -t "$HTX" -r "$HRX" -m 16 --num_runs="$NUM_RUNS" -v --export
python assign_frequencies.py -f "$FREQ0" -df 10e6 -n 300 -dmin 150 -dmax 200 -t "$HTX" -r "$HRX" -m 16 --num_runs="$NUM_RUNS" -v --export
python assign_frequencies.py -f "$FREQ0" -df 10e6 -n 300 -dmin 150 -dmax 200 -t "$HTX" -r "$HRX" -m 8 --num_runs="$NUM_RUNS" -v --export
python assign_frequencies.py -f "$FREQ0" -df 10e6 -n 300 -dmin 150 -dmax 200 -t "$HTX" -r "$HRX" -m 32 --num_runs="$NUM_RUNS" -v --export
python assign_frequencies.py -f "$FREQ0" -df 50e6 -n 60 -dmin 90 -dmax 100 -t "$HTX" -r "$HRX" -m 16 --num_runs="$NUM_RUNS" -v --export

python assign_frequencies.py -f "$FREQ0" -df 1e6 -n 400 -m 8 -dmin 90 -dmax 100 -t "$HTX" -r "$HRX" --num_runs="$NUM_RUNS" -v --export
python assign_frequencies.py -f "$FREQ0" -df 120e3 -n 3000 -m 8 -dmin 90 -dmax 100 -t "$HTX" -r "$HRX" --num_runs="$NUM_RUNS" -v --export
