# `pycvc`: Python Analysis Tools for the CVC Dataset

## Installation
The best way to install `pycvc` and get started is by through the Anaconda distribution of Python and its `conda` package manager.

### Download Anaconda
Anaconda is a free and open-source distribution of Python built and maintained by Continuum Analytics.
Windows, Mac OS X, and Linux users can download the Anaconda installers [here](http://continuum.io/downloads).

`conda` is a package manager bundled with Anaconda.
Package managers make it easy to install third party python libraries as well as other hard-to-install tools such as GDAL for spatial analysis and R for statistics.

Linux and Mac OS X users should follow the installation instructions
It is recommended to install and use the 64-bit version of Anaconda if your system allows.
However, `pycvc` should work with the 32-bit version without errors.

Running the Anaconda installer will not require administrative privileges to install.
If asked, it is recommended to select the options (when presented) to add Anaconda to the system path and to add it to the Windows registry (the later not being applicable on Max OS X and Linux).

After installing Anaconda, confirm that the installation was successful by opening a DOS-prompt or terminal and typing: `conda --version` and hit *enter*.
On Windows, the output will look something like this:
```
Microsoft Windows [Version 6.3.9600]
(c) 2013 Microsoft Corporation. All rights reserved.

C:\Users\phobson
$ conda --version
conda 3.16.0
```

Leave this prompt/terminal open as we will need it throughout the installation process.

## Setting up the `pycvc` environment
After successfully installing Anaconda, it's type to configure Anaconda and create a new "environment" for the CVC data analysis tools with custom python extensions.

### Configuration
Before beginning, we need to tell the `conda` package manager to look at our custom channel for our project-specific packages.
To do this, execute the command `conda config --add channels phobson`.
If this command is successful, not output will be printed to the prompt/terminal.

### Create a CVC-specific environment
Now we need to install `pycvc` and its dependencies inside an isolated `conda` environment.
We do this in an isolated environment to prevent dependency-conflicts with other software installed on the system.

To create a new environment for `pycvc`, execute the following command: `conda create --name=cvc python=3.4 jupyter nose`.
The system will prompt you to confirm that you would like to download and install Python version 3.4, and the latest versions of `jupyter` (for interactive code execution), `nose` (for running tests), and all of their dependencies into a new environment called `cvc`.
Hit "y" and then the *enter* key to confirm.
The prompt will keep track of the progress of the download and installation process.
After it has completed, the terminal should display something like the following:
```
Extracting packages ...
[      COMPLETE      ]|##################################################| 100%
Linking packages ...
[      COMPLETE      ]|##################################################| 100%
#
# To activate this environment, use:
# > activate cvc
#
```

### Activating the environment
Just as the prompt's directions imply, we must "activate" the environment to use.
To do so type `activate cvc` into the prompt and hit *enter*.
Note that on Mac OS X and Linux, the command is a slightly different `source activate cvc`.
A successful activation of the the `cvc` environment will slightly change the appearance of the prompt by prepending the environments name to the prompt:
```
C:\Users\phobson
$ activate cvc
Activating environment "cvc"...

[cvc] C:\Users\phobson
$ # notice the new prefix
```

The prefix is removed after deactivating to signal that the environment is no longer active.
```
[cvc] C:\Users\phobson
$ deactivate
Deactivating environment "cvc"...

C:\Users\phobson
$ # now the prefix is gone
```
Again, note that Mac OS X and Linux users will need to use `source deactivate`.

### Installing `pycvc` and the remaining dependencies
`pycvc` has three main dependencies:
  1. `wqio`
  2. `pybmp`
  3. `pynsqd`

All three of these libraries are open source, written/maintained by Geosyntec, and installable through `conda`.
To install them into the `cvc` environment, confirm that the `cvc` environment is active and use `conda install` in the following manner:
```
C:\Users\phobson
$ activate cvc
Activating environment "cvc"...

[cvc] C:\Users\phobson
$ conda install pycvc
```
The output of that command will look like the following:
```
Fetching package metadata: ......
Solving package specifications: ...........
Package plan for installation in environment C:\Users\phobson\AppData\Local\Continuum\Miniconda3\envs\cvc:

The following NEW packages will be INSTALLED:

    pybmp:  0.1dev-1
    pycvc:  0.1dev-1
    pynsqd: 0.1dev-1
    wqio:   0.1dev-1

Proceed ([y]/n)? y

Linking packages ...
[      COMPLETE      ]|##################################################| 100%
```
At that point, `pycvc` and all of its dependencies are installed.
