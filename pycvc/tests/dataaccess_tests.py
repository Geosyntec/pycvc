import sys
import os
from six import StringIO
import datetime

from nose.tools import *
import nose.tools as nt
import numpy as np
import numpy.testing as nptest
import pandas
import pandas.util.testing as pdtest
import pyodbc

import wqio
from wqio import utils

from  pycvc import dataAccess


class test__check_sampletype(object):
    def test_grab(self):
        nt.assert_equal(dataAccess._check_sampletype('GraB'), 'grab')

    def test_composite(self):
        nt.assert_equal(dataAccess._check_sampletype('comPOSite'), 'composite')

    @nt.raises(ValueError)
    def test_junk(self):
        dataAccess._check_sampletype('junk')


class test__checkrescol(object):
    def test_conc(self):
        rescol, unitscol = dataAccess._check_rescol('concenTRATION')
        nt.assert_equal(rescol, 'concentration')
        nt.assert_equal(unitscol, 'units')

    def test_load(self):
        rescol, unitscol = dataAccess._check_rescol('load_outflow')
        nt.assert_equal(rescol, 'load_outflow')
        nt.assert_equal(unitscol, 'load_units')

    @nt.raises(ValueError)
    def test_junk(self):
        dataAccess._check_rescol('junk')


class test__check_timegroup(object):
    def test_season(self):
        nt.assert_equal(dataAccess._check_timegroup('seASon'), 'season')

    def test_year(self):
        nt.assert_equal(dataAccess._check_timegroup('YEAR'), 'year')

    def test_grouped_season(self):
        nt.assert_equal(dataAccess._check_timegroup('grouped_season'), 'grouped_season')

    @nt.raises(ValueError)
    def test_junk(self):
        dataAccess._check_timegroup('junk')


class test__grouped_season(object):
    def setup(self):
        self.winter = '2012-02-02'
        self.spring = datetime.datetime(2008, 4, 1)
        self.summer = pandas.Timestamp("2013-07-21")
        self.autumn = '2001-11-24'

    def test_winter(self):
        nt.assert_equal(dataAccess._grouped_seasons(self.winter), 'winter/spring')

    def test_spring(self):
        nt.assert_equal(dataAccess._grouped_seasons(self.spring), 'winter/spring')

    def test_summer(self):
        nt.assert_equal(dataAccess._grouped_seasons(self.summer), 'summer/autumn')

    def test_autumn(self):
        nt.assert_equal(dataAccess._grouped_seasons(self.autumn), 'summer/autumn')

    @nt.raises(ValueError)
    def test_junk(self):
        dataAccess._grouped_seasons("junk")


class test__remove_storms_from_df(object):
    def setup(self):
        self.df = pandas.DataFrame({
            'storm_number': np.arange(20),
            'datecol1': pandas.date_range(start='2014-01-01', freq='16H', periods=20),
            'datecol2': pandas.date_range(start='2014-01-03', freq='12H', periods=20)
        })

        self.exclude_dates = [
            '2014-01-01',
            pandas.Timestamp('2014-01-02'),
            datetime.datetime(2014, 1, 3),
            '2014-01-10',
        ]

        self.known_datecol1_shape = (14, 3)
        self.known_datecol2_shape = (16, 3)

    def test_datecol1(self):
        df = dataAccess._remove_storms_from_df(self.df, self.exclude_dates, 'datecol1')
        nt.assert_tuple_equal(df.shape, self.known_datecol1_shape)

    def test_datecol2(self):
        df = dataAccess._remove_storms_from_df(self.df, self.exclude_dates, 'datecol2')
        nt.assert_tuple_equal(df.shape, self.known_datecol2_shape)

