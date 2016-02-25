import sys
import os
from six import StringIO
import datetime
from pkg_resources import resource_filename

import nose.tools as nt
from nose.plugins.attrib import attr
import numpy as np
import numpy.testing as nptest
import pandas
import pandas.util.testing as pdtest
import pyodbc

import wqio
from wqio import utils

from  pycvc import dataAccess, external


def test_fix_cvc_bacteria_units():
    df = pandas.DataFrame({
        'UOM': ['mg/L', 'ft', 'CFU/100mL', 'CFU/100 mL'],
    })

    expected = pandas.DataFrame({
        'UOM': ['mg/L', 'ft', 'CFU/100 mL', 'CFU/100 mL'],
    })

    result = dataAccess._fix_cvc_bacteria_units(df, unitscol='UOM')
    pdtest.assert_frame_equal(result, expected)


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


class test_database_no_ext(object):
    def setup(self):
        dbfile = resource_filename("pycvc.tests.testdata", "test.accdb")
        self.db = cvcdb = dataAccess.Database(dbfile)

    def test_connect(self):
        with self.db.connect() as cnn:
            nt.assert_true(isinstance(cnn, pyodbc.Connection))

    def test_sites(self):
        cols = ['site', 'sitename', 'total_area',
                'impervious_area', 'bmp_area',
                'weir_height_mm']
        expected = pandas.DataFrame({
            'bmp_area': {0: 675.0, 1: 0.0, 2: 541.0, 3: 336.0},
            'impervious_area': {0: 2578.0, 1: 4632.0, 2: 5838.0, 3: 1148.0},
            'site': {0: 'ED-1', 1: 'LV-1', 2: 'LV-2', 3: 'LV-4'},
            'sitename': {
                0: 'Elm Drive', 1: 'Lakeview Curb and Gutter',
                2: 'Lakeview Grass Swales',
                3: 'Lakeview Bioswales and Permeable Pavement'
            },
            'total_area': {0: 5781.0, 1: 17799.0, 2: 16962.0, 3: 3785.0},
            'weir_height_mm': {0: 155, 1: 165, 2: 152, 3: 490},
        })[cols]

        pdtest.assert_frame_equal(expected, self.db.sites)

    def test__check_site(self):
        nt.assert_equal('ED-1', self.db._check_site('ED-1'))
        nt.assert_raises(ValueError, self.db._check_site, 'junk')

    def test_getWQData(self):
        ed1 = self.db.getWQData('ED-1')
        nt.assert_true(isinstance(ed1, pandas.DataFrame))
        nt.assert_tuple_equal(ed1.shape, (12, 11))

        lv1 = self.db.getWQData('LV-1')
        nt.assert_true(isinstance(lv1, pandas.DataFrame))
        nt.assert_tuple_equal(lv1.shape, (0, 11))

    def test_getHydroData(self):
        ed1 = self.db.getHydroData('ED-1')
        nt.assert_true(isinstance(ed1, pandas.DataFrame))
        nt.assert_tuple_equal(ed1.shape, (8784, 2))
        nt.assert_true(isinstance(ed1.index, pandas.DatetimeIndex))

    def test_getDrainageArea(self):
        da = self.db.getDrainageArea('ED-1')
        nt.assert_true(isinstance(da, wqio.DrainageArea))
        nt.assert_equal(da.total_area, 5781)
        nt.assert_equal(da.bmp_area, 675)
        nt.assert_equal(da.imp_area, 2578)

    def test_getSamples(self):
        df = self.db.getSamples('ED-1')
        nt.assert_true(isinstance(df, pandas.DataFrame))
        nt.assert_tuple_equal(df.shape, (6, 11))

    @attr(speed='slow')
    def test_wqstd(self):
        nt.assert_true(isinstance(self.db.wqstd, pandas.DataFrame))
        nt.assert_tuple_equal(self.db.wqstd.shape, (48, 4))
        nt.assert_list_equal(
            ['parameter', 'units', 'lower_limit', 'upper_limit'],
            self.db.wqstd.columns.tolist()
        )
