# Frequency Assignment for Guaranteed QoS in Two-Ray Models with Limited Location Information

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klb2/two-ray-freq-assignment/HEAD)
![GitHub](https://img.shields.io/github/license/klb2/two-ray-freq-assignment)


This repository is accompanying the paper "Frequency Assignment for Guaranteed
QoS in Two-Ray Models with Limited Location Information" (K.-L. Besser, E.
Jorswieck, J. Coon, H. V. Poor. May 2025, WiOpt 2025).

The idea is to give an interactive version of the calculations and presented
concepts to the reader. One can also change different parameters and explore
different behaviors on their own.


## File List
The following files are provided in this repository:

- `run.sh`: Bash script that reproduces the figures presented in the paper.
- `assign_frequencies.py`: Python module that contains the implementations of
  the frequency assignment algorithms.
- `example_local_minima.py`: Python script that shows the numerical example of
  calculating the distances where destructive interference occurs.
- `model.py`: Python module that contains functions about the system model,
  e.g., to calculate the lengths of the paths.
- `multiple_frequencies.py`: Python module that contains the functions to
  calculate the receive power when multiple frequencies are used in parallel.
- `roots_fourier_series.py`: Python module that contains functions to calculate
  the zeros of a Fourier series.
- `single_frequency.py`: Python module that contains the functions to calculate
  the receive power when a single frequency is used.
- `util.py`: Python module that contains utility functions, e.g., for saving results.


## Usage
### Running it online
The easiest way is to use services like [Binder](https://mybinder.org/) or
[CodeOcean](https://codeocean.com/) to run the scripts online without having to
install anything locally.

### Local Installation
If you want to run it locally on your machine, Python3 and some libraries are
needed.
The present code was developed and tested with the following versions:

- Python 3.13
- numpy 2.2
- scipy 1.15
- pandas 2.2

Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages by running
```bash
pip3 install -r requirements.txt
```
This will install all the needed packages which are listed in the requirements 
file.

You can then recreate the figures and simulation results from the paper by
running
```bash
bash run.sh
```


## Acknowledgements
This research was supported by the German Research Foundation (DFG) under grant
BE 8098/1-1, by the Federal Ministry of Education and Research Germany (BMBF)
as part of the 6G Research and Innovation Cluster 6G-RIC under Grant 16KISK031,
by the EPSRC under grant number EP/T02612X/1, and by the U.S National Science
Foundation under Grants CNS-2128448 and ECCS-2335876.


## License and Referencing
This program is licensed under the MIT license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```bibtex
@misc{Besser2025frequency,
  author = {Besser, Karl-Ludwig and Jorswieck, Eduard A. and Coon, Justin P. and Poor, H. Vincent},
  title = {Frequency Assignment for Guaranteed {QoS} in Two-Ray Models with Limited Location Information},
	booktitle = {23rd International Symposium on Modeling and Optimization in Mobile, Ad hoc, and Wireless Networks (WiOpt)},
	year = {2025},
	month = {5},
	publisher = {IEEE},
}
```
