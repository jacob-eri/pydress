# pydress
Directional Relativistic Spectrum Simulator (DRESS), mainly developed for fusion reactions.

## Installation
The package is installed by the following sequence of steps:

* Download the code from the git repository (https://github.com/jacob-eri/pydress).
* Unpack the code where you like. In what follows we assume that it was unpacked in "/foo/bar/pydress"
* Go into "/foo/bar/pydress/dress" and open `config.py`.
* Change the variable `cross_section_dir` so that it contains the absolute path to the "cross-section" folder (this folder is found in "/foo/bar/pydress/data", but you may of course put it anywhere you like on your system).
* Go to "/foo/bar/pydress" and run the install script in the usual way, e.g. 

`python setup.py install`

Or, if you want to install locally for your own user

`python setup.py install --user`

## Documentation
A short Jupyter notebook demonstrating the basic capabilities of the package can be found in "/foo/bar/pydress/doc".
