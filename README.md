visid
=====

Variational System Identification using Bayesian Networks.

Development and Testing
=======================

For development and test of the package, a recent Ubuntu distribution is
recommended. On Ubuntu 22.04, the following dependencies are needed.

```
sudo apt install python3.10 python3.10-venv git 
```

Clone the repository and enter into it.

```
git clone git@github.com:dimasad/visid.git
cd visid
```

The creation of a virtual environment is recommended.

```
python3.10 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
```

For installing this package with CPU-only support, run the command below.

```
pip install -e .[cpu,devextra]
```

Alternatively, for installing this package with GPU and CPU support, run the
commands below.

```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .[devextra]
```
