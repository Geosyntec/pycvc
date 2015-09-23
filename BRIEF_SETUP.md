
# Brief setup instructions

## Dependencies
### Required
 * [Anaconda installers](http://continuum.io/downloads)
 * Microsoft Access or [Access Database Engine](http://www.microsoft.com/en-us/download/details.aspx?id=13255)
 * [ODBC drivers](http://www.microsoft.com/en-us/download/details.aspx?id=13255).

### Optional
 * [MiKTeX](http://miktex.org/download)
 * [git](https://help.github.com/articles/set-up-git/).

## Configuring the `conda` environment

```
conda config --add channels phobson
conda create --name=cvc python=3.4 jupyter nose pycvc
```

## Starting the notebook server to use `pycvc`
```
activate cvc
ipython notebook
```

## Installing from an editable source directory
```
git clone https://github.com/Geosyntec/pycvc.git
cd pycvc
activate cvc
pip install -e .
```
