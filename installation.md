# Installation
Installation of `python-allib` requires `Python 3.8` or higher.

### 1. Python installation
Install Python on your operating system using the [Python Setup and Usage](https://docs.python.org/3/using/index.html) guide.

### 2. Installing `python-allib`
`python-allib` can be installed:

* _using_ `pip`: `pip3 install` (released on [PyPI](https://pypi.org/project/python-allib/))
* _locally_: cloning the repository and using `python setup.py install` (NB. On Ubuntu, you may need to use python3 if python is not available or refers to Python 2.x).

#### Using `pip`
1. Open up a `terminal` (Linux / macOS) or `cmd.exe`/`powershell.exe` (Windows)
2. Run the command:
    - `pip install python-allib`, or
    - `pip install python-allib`.

```console
user@terminal:~$ pip3 install python-allib
Collecting python-allib
...
Installing collected packages: python-allib
Successfully installed python-allib
```

#### Locally
1. Download the folder from `GitLab/GitHub`:
    - Clone this repository, or 
    - Download it as a `.zip` file and extract it.
2. Open up a `terminal` (Linux / macOS) or `cmd.exe`/`powershell.exe` (Windows) and navigate to the folder you downloaded `python-allib` in.
3. In the main folder (containing the `setup.py` file) run:
    - `python3 setup.py install`, or
    - `python setup.py install`.

```console
user@terminal:~$ cd ~/python-allib
user@terminal:~/python-alliby$ python3 setup.py install
running install
running bdist_egg
running egg_info
...
Finished processing dependencies for python-allib
```
### 3. Installing `R` for Chao's Estimator (Optional)

For the usage of Rivest's version of Chao's estimator, a working `R` installation is needed.
Furthermore, the following `R` packages are needed.

- `tidyverse`
- `RCapture`

These can be installed in the R console with the following command.

```r
install.packages(c("tidyverse", "RCapture"))
```
Besides these two `R` packages, the `rpy2` python package is needed (install it using `pip install rpy2`).