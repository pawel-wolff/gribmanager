# gribmanager

An object-oriented Python wrapper around (a part) of ECMWF's ecCodes library.


## Description

Reads GRIB data format and performs 3D interpolation, capable of handling IFS' hybrid vertical coordinates.

TODO (refer to notebooks which show typical use cases: TBD)

> `tests/*.py` are outdated and needs some adaptations to make them run 


## Installation

> It is highly recommended to use a **virtual environment** for the installation
(`conda`, `venv`, etc.) instead of a *system-wide environment*.


### Install the package directly from a git repository 

If you have [configured ssh keys to your gitlab account](https://docs.gitlab.com/ee/user/ssh.html#add-an-ssh-key-to-your-gitlab-account),
use
```sh
python -m pip install git+ssh://git@gitlab.in2p3.fr/sedoo/iagos/pva/core/gribmanager.git
```
Otherwise, use
```sh
python -m pip install git+https://gitlab.in2p3.fr/sedoo/iagos/pva/core/gribmanager.git
```
For updating the package from git, use `--force-reinstall --no-deps` options, e.g.
```sh
python -m pip install --force-reinstall --no-deps git+ssh://git@gitlab.in2p3.fr/sedoo/iagos/pva/core/gribmanager.git
```


### Install the package in the development mode

First, clone this git repository doing
```sh
git clone git@gitlab.in2p3.fr:sedoo/iagos/pva/core/gribmanager.git 
```
or
```sh
git clone https://gitlab.in2p3.fr/sedoo/iagos/pva/core/gribmanager.git
```
Next, install the package from the local repository in the *editable mode*:
```sh
cd gribmanager
python -m pip install --editable .
```
Thanks to the editable mode, changes to your local repository will be visible in all new python sessions. 

> Before installing the package with `pip`, you can enable dry-run mode by adding
the `--dry-run` option (e.g., `python -m pip install --editable . --dry-run`). 
This mode allows you to identify any required packages that are not yet installed 
in your environment. 
> 
> It is recommended to first install these dependencies using 
your environment manager (e.g., `conda`) and then proceed with the package installation 
as shown above (either from `git` or in development mode from a local repository). 
While `pip` can automatically install any missing dependencies, doing so may 
result in these dependencies not being managed by your environment manager in the future.