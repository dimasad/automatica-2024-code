Code for paper "Parameterizations for Large-Scale Variational System 
Identification Using Unconstrained Optimization"

[DOI: 10.1016/j.automatica.2024.112086](https://doi.org/10.1016/j.automatica.2024.112086)

Instructions for Running Code
=============================

The code has been tested on Ubuntu 22.04 but should work on any system with
python 3.10+ and JAX/FLAX. The instructions below assume a Ubuntu 22.04 distro,
possibly in WSL.

Install dependencies
--------------------

On Ubuntu 22.04, the following dependencies are needed.

```bash
sudo apt install python3.10 python3.10-venv git bc
```
Clone repository
----------------

Clone the repository and enter into it.

```bash
git clone https://github.com/dimasad/automatica-2024-code.git
cd automatica-2024-code
```

Setup Python Environment
------------------------

The creation of a virtual environment is recommended.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
```

Install jax with GPU support for the benchmarks. And this package with the 
variational inference library.

```bash
pip install -U "jax[cuda12]==0.4.38"
pip install -e .[devextra]
```

Run Experiments
---------------

Each example is an executable bash script. If SLURM is installed it will run as
in WVU Research Computing Thorny Flat cluster, otherwise it will just run all
scripts in order (which takes a loooooooooong time). See the bash scripts for
how to run each test separately. 

There is a run script and a job submission script which calls the underlying run
with all parameters tested. Please contact me if you need help running this. My
e-mail is in the paper.

```bash
scripts/duffing-vsi-submit
python scripts/linsys-ssest-submit.py
scripts/linsys-timeit-submit
scripts/linsys-vsi-submit
```
