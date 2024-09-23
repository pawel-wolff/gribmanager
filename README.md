# gribmanager

An object-oriented Python wrapper around (a part) of ECMWF's ecCodes library.


## Description

Reads GRIB data format and performs 3D interpolation, capable of handling IFS' hybrid vertical coordinates.


## Installation

### Clone the git repository

```sh
git clone https://github.com/pawel-wolff/gribmanager.git
cd gribmanager
```

### Install the package

```sh
python -m pip install .
```

### Install the package in the development mode

```sh
python -m pip install --editable .
```

### Test your installation (requires a sample GRIB file)

```sh
python tests/test_gribmanager.py
```
