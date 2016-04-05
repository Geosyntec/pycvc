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

from  pycvc import dataAccess


ON_WINDOWS = sys.platform == 'win32'


@nt.nottest
def load_test_data(filename, **opts):
    return pandas.read_csv(resource_filename('pycvc.tests.testdata', filename), **opts)


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

    def test_with_scalar(self):
        df = dataAccess._remove_storms_from_df(self.df, self.exclude_dates[0], 'datecol1')
        known_shape = (18, 3)
        nt.assert_tuple_equal(df.shape, known_shape)


class test_Database_no_externals(object):
    def setup(self):
        dbfile = resource_filename("pycvc.tests.testdata", "test.accdb")
        self.db = dataAccess.Database(dbfile)

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_connect(self):
        with self.db.connect() as cnn:
            nt.assert_true(isinstance(cnn, pyodbc.Connection))

    @nptest.dec.skipif(not ON_WINDOWS)
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

    @nptest.dec.skipif(not ON_WINDOWS)
    def test__check_site(self):
        nt.assert_equal('ED-1', self.db._check_site('ED-1'))
        nt.assert_raises(ValueError, self.db._check_site, 'junk')

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_getWQData(self):
        ed1 = self.db.getWQData('ED-1')
        nt.assert_true(isinstance(ed1, pandas.DataFrame))
        nt.assert_tuple_equal(ed1.shape, (12, 11))

        lv1 = self.db.getWQData('LV-1')
        nt.assert_true(isinstance(lv1, pandas.DataFrame))
        nt.assert_tuple_equal(lv1.shape, (0, 11))

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_getHydroData(self):
        ed1 = self.db.getHydroData('ED-1')
        nt.assert_true(isinstance(ed1, pandas.DataFrame))
        nt.assert_tuple_equal(ed1.shape, (39600, 2))
        nt.assert_true(isinstance(ed1.index, pandas.DatetimeIndex))

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_getDrainageArea(self):
        da = self.db.getDrainageArea('ED-1')
        nt.assert_true(isinstance(da, wqio.DrainageArea))
        nt.assert_equal(da.total_area, 5781)
        nt.assert_equal(da.bmp_area, 675)
        nt.assert_equal(da.imp_area, 2578)

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_getSamples(self):
        df = self.db.getSamples('ED-1')
        nt.assert_true(isinstance(df, pandas.DataFrame))
        nt.assert_tuple_equal(df.shape, (6, 11))

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_getRatingCurve(self):
        df = self.db.getRatingCurve('ED-1')
        nt.assert_true(isinstance(df, pandas.DataFrame))
        nt.assert_tuple_equal(df.shape, (363, 2))

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_wqstd(self):
        nt.assert_true(isinstance(self.db.wqstd, pandas.DataFrame))
        nt.assert_tuple_equal(self.db.wqstd.shape, (48, 4))
        nt.assert_list_equal(
            ['parameter', 'units', 'lower_limit', 'upper_limit'],
            self.db.wqstd.columns.tolist()
        )


class test_Site(object):
    def setup(self):
        dbfile = resource_filename("pycvc.tests.testdata", "test.accdb")
        self.db = dataAccess.Database(dbfile)

        influentcsv = StringIO(textwrap.dedent("""\
        parameter,units,season,influent lower,influent median,influent upper
        Cadmium (Cd),ug/L,autumn,0.117,0.361,0.55
        Cadmium (Cd),ug/L,spring,0.172,0.352,0.53
        Cadmium (Cd),ug/L,summer,0.301,0.411,0.46
        Cadmium (Cd),ug/L,winter,0.355,0.559,1.125
        Lead (Pb),ug/L,autumn,6.173,10.9,16.25
        Lead (Pb),ug/L,spring,8.6,19.9,28.0
        Lead (Pb),ug/L,summer,7.69,17.9,22.6
        Lead (Pb),ug/L,winter,9.0,27.0,47.5
        """))
        self.influent = pandas.read_csv(influentcsv)

        self.runoff_fxn = lambda r: 10**(1.58 + 0.02*r['total_precip_depth'])
        self.bypass_fxn = lambda r: max(0, 1.22 * r['total_precip_depth'])
        self.inflow_fxn = lambda r: max(0, self.runoff_fxn(r) - self.bypass_fxn(r))

        self.site = dataAccess.Site(
            db=self.db, siteid='ED-1', raingauge='ED-1',
            influentmedians=self.influent, color='b', marker='s',
            tocentry='Elm Drive', runoff_fxn=self.runoff_fxn,
            bypass_fxn=self.bypass_fxn, inflow_fxn=self.inflow_fxn,
        )

    def test_influent_medians(self):
        pdtest.assert_frame_equal(self.site.influentmedians, self.influent)

    def test_hydro_functions(self):
        nt.assert_equal(self.site.runoff_fxn, self.runoff_fxn)
        nt.assert_equal(self.site.bypass_fxn, self.bypass_fxn)
        nt.assert_equal(self.site.inflow_fxn, self.inflow_fxn)

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_wqstd(self):
        pdtest.assert_frame_equal(self.site.wqstd, self.db.wqstd)

    def test__rating_curve_data(self):
        nt.assert_true(isinstance(self.site._rating_curve_data, pandas.DataFrame))
        nt.assert_list_equal(
            self.site._rating_curve_data.columns.tolist(),
            ['head_mm', 'flow_lps']
        )

    def test_drainage_area(self):
        nt.assert_equal(self.site.drainagearea.simple_method(10), 32842.5)

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_wqdata(self):
        expected_columns = ['site', 'sample', 'sampletype', 'samplestart', 'samplestop',
                            'interval_minutes', 'parameter', 'units', 'detectionlimit',
                            'qualifier', 'concentration', 'season', 'grouped_season', 'year']
        expected_shape = (12, 14)

        nt.assert_list_equal(self.site.wqdata.columns.tolist(), expected_columns)
        nt.assert_tuple_equal(self.site.wqdata.shape, expected_shape)
        self.site.wqdata.to_csv("pycvc/tests/testdata/baseline_wqdata.csv", index=False)

    def test_hydrodata(self):
        expected = load_test_data('baseline_hydrodata.csv')
        pdtest.assert_frame_equal(
            self.site.hydrodata.data.reset_index(drop=True), expected,
            check_dtype=False
        )

    def test_sample_info(self):
        datecols = ['samplestart', 'samplestop', 'collectiondate']
        expected = load_test_data('baseline_sample_info.csv', parse_dates=datecols)
        self.site.sample_info['samplestart']
        pdtest.assert_frame_equal(self.site.sample_info, expected)

    def test_grabdates(self):
        dates = pandas.Series(map(
            pandas.Timestamp,
            ['2012-07-31 12:20:00', '2012-08-10 03:20:00', '2012-08-11 11:37:00'
        ]), index=[0, 2, 4], name='samplestart')
        pdtest.assert_series_equal(self.site.grabdates, dates)

    def test_compdates(self):
        dates = pandas.Series(map(
            pandas.Timestamp,
            ['2012-07-31 12:24:00', '2012-08-10 03:24:00', '2012-08-11 11:41:00'
        ]), index=[1, 3, 5], name='samplestart')
        pdtest.assert_series_equal(self.site.compdates, dates)

    def test_compendtimees(self):
        dates = pandas.Series(map(
            pandas.Timestamp,
            ['2012-07-31 16:45:00', '2012-08-10 20:54:00', '2012-08-11 17:31:00'
        ]), index=[1, 3, 5], name='samplestop')
        pdtest.assert_series_equal(self.site.compendtimes, dates)

    def test_max_flow(self):
        nt.assert_equal(self.site.max_flow, 10.957)

    def test_max_inflow(self):
        nt.assert_true(np.isnan(self.site.max_inflow))

    def test_sampled_storms(self):
        with mock.patch.object(self.site, '_get_storms_with_data') as _gsd:
            _ = self.site.sampled_storms
            _gsd.assert_called_once_with(sampletype='composite')

    def test_unsampled_storms(self):
        with mock.patch.object(self.site, '_get_storms_without_data') as _gsd:
            _ = self.site.unsampled_storms
            _gsd.assert_called_once_with(sampletype='composite')

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_storms(self):
        nt.assert_true(isinstance(self.site.storms, dict))
        nt.assert_equal(len(self.site.storms), 27)
        for sn in self.site.storms:
            nt.assert_true(isinstance(self.site.storms[sn], wqio.Storm))

    def test_all_samples(self):
        nt.assert_true(isinstance(self.site.all_samples, list))
        nt.assert_equal(len(self.site.all_samples), 6)
        for s in self.site.all_samples:
            nt.assert_true(isinstance(s, wqio.GrabSample) or isinstance(s, wqio.CompositeSample))

    def test_sample(self):
        nt.assert_list_equal(
            sorted(list(self.site.samples.keys())),
            ['composite', 'grab']
        )

        for cs in self.site.samples['composite']:
            nt.assert_true(isinstance(cs, wqio.CompositeSample))

        for gs in self.site.samples['grab']:
            nt.assert_true(isinstance(gs, wqio.GrabSample))

    @nptest.dec.skipif(not ON_WINDOWS)
    def test_storm_info(self):
        expected = load_test_data('baseline_storm_info.csv', parse_dates=['start_date', 'end_date'])
        expected = expected.assign(year=expected['year'].astype(str))
        pdtest.assert_frame_equal(self.site.storm_info, expected)
