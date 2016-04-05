import sys
import os
from six import StringIO
import datetime
from pkg_resources import resource_filename
import textwrap
from io import StringIO

import nose.tools as nt
from nose.plugins.attrib import attr
from unittest import mock
import numpy.testing as nptest
import pandas.util.testing as pdtest


import numpy as np
import pandas
import pyodbc

import wqio
from wqio import utils

from  pycvc import dataAccess, external


def test__fix_nsqd_bacteria_units():
    cols = ['param', 'conc_units', 'res']
    inputdf = pandas.DataFrame({
        'conc_units': ['MPN/100 mL', 'MPN/100 mL', 'CFU/100 mL', 'ug/L'],
        'param': ['E Coli', 'E Coli', 'Fecal', 'Copper'],
        'res': [1, 2, 3, 4]
    })

    outputdf = external._fix_nsqd_bacteria_units(inputdf, unitscol='conc_units')
    expected = pandas.DataFrame({
        'conc_units': ['CFU/100 mL', 'CFU/100 mL', 'CFU/100 mL', 'ug/L'],
        'param': ['E Coli', 'E Coli', 'Fecal', 'Copper'],
        'res': [1, 2, 3, 4]
    })

    pdtest.assert_frame_equal(outputdf[cols], expected[cols])
