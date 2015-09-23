# `pycvc`: Python Analysis Tools for the CVC Dataset

## Quick-start guides
Advanced users can go straight to the [condensed version of this document](https://github.com/Geosyntec/pycvc/blob/master/BRIEF_SETUP.md).
New users should continue reading.

## Installation
The best way to install `pycvc` and get started is by through the Anaconda distribution of Python and its `conda` package manager.

### Python - Download Anaconda
Anaconda is a free and open-source distribution of Python built and maintained by Continuum Analytics.
Windows, Mac OS X, and Linux users can download the Anaconda installers [here](http://continuum.io/downloads).

`conda` is a package manager bundled with Anaconda.
Package managers make it easy to install third party python libraries as well as other hard-to-install tools such as GDAL for spatial analysis and R for statistics. For a more detailed description of conda, check out the [official documentation](http://conda.pydata.org/docs/)

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

### Non-Python Dependencies
#### Microsoft Access
There are a few additional software requirements related to accessing the data from python.
The first is of course, a modern version Microsoft Access that supports the ".accdb" file format.
If this software is unavailable, it *may* be possible to scrape by with the redistributable [Access Database Engine](http://www.microsoft.com/en-us/download/details.aspx?id=13255).

#### ODBC Drivers
The second software dependency is an Open Database Connectivity (ODBC) driver that supports your version of Microsoft Access.
Typically, these drivers are installed along with Microsoft Office.
If they are not, you can download them [directly from Microsoft](http://www.microsoft.com/en-us/download/details.aspx?id=13255).

#### LaTeX Implementation (optional)
On Windows, [MiKTeX](http://miktex.org/download) can optionally be used to compile LaTeX documents and manage third-party LaTeX extensions.
Thus installing MiKTeX and [adding it to the system path](http://www.howtogeek.com/118594/how-to-edit-your-system-path-for-easy-command-line-access/) will enable `pycvc` to compile the Individual Storm Reports (ISRs) into stand-alone PDFs.
If `pycvc` cannot find the required executables that are installed with MiKTeX (e.g., `pdflatex`), `pycvc` will issue a non-fatal warning alerting the user.
In such a case, all of the tables and figures that comprise ISRs will be generated, but as individual CSV and PNG files.

#### Source code management
Directly downloading and managing the source code of this project is best done through software known as git.
Github itself has very thorough [documentation on setting up and use git](https://help.github.com/articles/set-up-git/).
See also the [contributing file](https://github.com/Geosyntec/pycvc/blob/master/CONTRIBUTING.md) for more links to information on working with python source code.

## Setting up the `pycvc` environment
After successfully installing Anaconda and the other dependencies, it's time to configure Anaconda and create a new "environment" for the CVC data analysis tools with custom python extensions.

### Configuration
Before beginning, we need to tell the `conda` package manager to look at our custom channel for our project-specific packages.
To do this, execute the following command:
```
conda config --add channels phobson
```

If this command is successful, no output will be printed to the prompt/terminal.

### Create a CVC-specific environment
Now we need to install `pycvc` and its dependencies inside an isolated `conda` environment.
We do this in an isolated environment to prevent dependency-conflicts with other software installed on the system.

To create a new environment for `pycvc`, execute the following command:
```
conda create --name=cvc python=3.4 jupyter nose
```

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
  2. `pybmpdb`
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

    pybmpdb:  0.1-1
    pycvc:  0.1-1
    pynsqd: 0.1-1
    wqio:   0.1-1

Proceed ([y]/n)? y

Linking packages ...
[      COMPLETE      ]|##################################################| 100%
```
At that point, `pycvc` and all of its dependencies are installed.

### Updating python libraries through conda
If the library packages uploaded to the conda servers are updated, you can update your local installation with the following commands:
```
C:\Users\phobson
$ activate cvc
Activating environment "cvc"...

[cvc] C:\Users\phobson
$ conda update --all
```


## Using `pycvc`
With `pycvc` and all of its dependencies installed, they best way to get started is by downloading and using the [notebook](https://github.com/Geosyntec/pycvc/blob/master/examples/Data%20Summaries.ipynb]) in this repository.

### Getting CVC Analysis Notebooks (and Source Code)
To do so, you can either use git to clone the entire repository and source code with the cmd:
```
git clone https://github.com/Geosyntec/pycvc.git
```
If cloning through git is not an option, you can download a zip file of everything using the buttons to the right of this webpage.

### Starting a Jupyter Notebook
Once you've downloaded and unzipped the repository to a convenient place, navigate to the `examples` directory in a command prompt, activate the `cvc` conda environment, and start a Jupyer/IPython notebook server. That will look *something* like this:

```
Microsoft Windows [Version 6.3.9600]
(c) 2013 Microsoft Corporation. All rights reserved.

C:\Users\phobson
$ cd Downloads\pycvc-master\pycvc-master\examples

C:\Users\phobson\Downloads
$ activate cvc
Activating environment "cvc"...

[cvc] C:\Users\phobson\Downloads
$ ipython notebook
[I 09:04:51.902 NotebookApp] Serving notebooks from local directory: C:\Users\phobson\Downloads
[I 09:04:51.902 NotebookApp] 0 active kernels
[I 09:04:51.902 NotebookApp] The IPython Notebook is running at: http://localhost:8888/
[I 09:04:51.902 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```
At that point, your default web browser (preferably not Internet Explorer) will pop up.
If that browswer happens to be internet explorer, it is recommended to copy the notebook server's URL into a modern browser like Chrome or Firefox.
Then, click the notebook in the list of files and a new browser tab will open with the analysis ready to go.

For more information about Jupyter, check out the [official documentation](http://jupyter.readthedocs.org/en/latest/)

### Configuring the Analysis to use your local database
The seventh code cell in the notebook titled **Load CVC Database** must be configured to point to your local copy of the CVC database. By default, the code cell is configured as follows:
```
cvcdbfile = "C:/users/phobson/Desktop/cvc.accdb"
cvcdb = pycvc.Database(cvcdbfile, nsqdata, bmpdb, testing=False)
```
Modifying the first line to include the path your copy of the database will enable the remainder of the code cells to run.
Be sure to separate the directories in path with slashes (`/`) instead of the typical backslashes (`\`) found on Windows.
As an example, `cvcdbfile = "C:/data/cvc.accdb"` will work, but `cvcdbfile = "C:\data\cvc.accdb"` will fail.

## Installing from source
It is recommended that you use `pip` if you wish to install `pycvc` from source.
To set that up, first clone the repository as discussed above, and then use `pip install`
```
git clone https://github.com/Geosyntec/pycvc.git
cd pycvc
activate cvc
pip install .  # the "." means the current directory
```

The above procedure copies and compiles deep inside your python's installation folders.
Alternatively, if you want to modify the source code and conveniently test out the changes you've made, add and `-e` (for editable) flag to the install command:
```
pip install -e .
```
When you do that, python simply creates a link in its installation directory to your clone repository.
So any modification you make to the source code will be reflected every time you restart your python interpreter (the jupyer notebook, most likely).
