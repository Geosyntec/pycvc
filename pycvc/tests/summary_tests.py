from io import StringIO
from textwrap import dedent
from datetime import date
from pkg_resources import resource_filename

import nose.tools as nt
import numpy as np
import numpy.testing as nptest
import pandas
import pandas.util.testing as pdtest

from  pycvc import summary


@nt.nottest
def csv_string_as_df(csv_string, **opts):
    return pandas.read_csv(StringIO(dedent(csv_string)), **opts)

@nt.nottest
def load_test_data(filename, **opts):
    return pandas.read_csv(resource_filename('pycvc.tests.testdata', filename), **opts)


def test_classify_storms():
    input_df = pandas.DataFrame({
        'A': np.arange(2, 29, 6),
        'B': np.arange(0, 41, 10),
    })

    expected_df = pandas.DataFrame({
        'A': np.arange(2, 29, 6),
        'B': np.arange(0, 41, 10),
        'C': ['<5 mm', '5 - 10 mm', '10 - 15 mm', '15 - 20 mm', '>25 mm'],
    })
    expected_df['C'] = expected_df['C'].astype('category')

    result_df = summary.classify_storms(input_df, 'A', newcol='C')

    pdtest.assert_frame_equal(result_df, expected_df)


def test_prevalence_table():
    input_df = csv_string_as_df("""\
    site,parameter,season,rescol,sampletype
    A,pA,Fall,1,composite
    A,pA,Fall,1,composite
    A,pA,Summer,1,composite
    A,pA,Summer,1,composite
    A,pA,Spring,1,composite
    A,pA,Winter,1,composite
    B,pA,Summer,1,composite
    B,pA,Fall,1,composite
    B,pA,Fall,1,composite
    B,pB,Spring,1,composite
    B,pB,Winter,1,grab
    B,pB,Winter,1,composite
    """)

    expected_df = pandas.DataFrame({
        'site': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'season': ['Fall', 'Spring', 'Summer', 'Winter'] * 2,
        'pA': [2, 1, 2, 1, 2, np.nan, 1, np.nan],
        'pB': [np.nan, np.nan, np.nan, np.nan, np.nan, 1, np.nan, 1],
    })[['site', 'season', 'pA', 'pB']]
    expected_df.columns.names = ['parameter']
    result_df = summary.prevalence_table(input_df, rescol='rescol', groupby_col='season')
    pdtest.assert_frame_equal(result_df, expected_df)


def test_remove_load_data_from_storms():
    input_df = pandas.DataFrame({
        'date': [
            pandas.Timestamp('2016-01-02 12:30'),
            pandas.Timestamp('2016-01-02 14:30'),
            pandas.Timestamp('2016-01-03 12:30'),
        ],
        'conc': [1., 2., 3.],
        'load': [1., 2., 4.],
        'load_a': [1., 2., 5.],
        'load_b': [1., 2., 6.]
    })

    expected_df = pandas.DataFrame({
        'date': [
            pandas.Timestamp('2016-01-02 12:30'),
            pandas.Timestamp('2016-01-02 14:30'),
            pandas.Timestamp('2016-01-03 12:30'),
        ],
        'conc': [1., 2., 3.],
        'load': [1., 2., 4.],
        'load_a': [np.nan, np.nan, 5],
        'load_b': [np.nan, np.nan, 6]
    })

    result_df = summary.remove_load_data_from_storms(input_df, [date(2016, 1, 2)], 'date')
    pdtest.assert_frame_equal(result_df, expected_df)


def test_pct_reduction():
    input_df = pandas.DataFrame({
        'A': [75., 50., 25., 10.],
        'B': [50., 50., 10., 20.],
    })

    expected_ser = pandas.Series([33.3333333333, 0., 60., -100.])

    result_ser = summary.pct_reduction(input_df, 'A', 'B')

    pdtest.assert_series_equal(result_ser, expected_ser)


def test_load_reduction_pct():
    input_df = pandas.DataFrame({
        'A':    np.array([75., 50., 25., 10.]),
        'A_lo': np.array([70., 45., 20.,  5.]),
        'A_hi': np.array([80., 55., 30., 15.]),
        'B':    np.array([50., 50., 10., 20.]),
        'B_lo': np.array([45., 45.,  5., 15.]),
        'B_hi': np.array([55., 55., 15., 25.]),
        'site': ['A', 'B'] * 2,
        'parameter': ['A'] * 4,
        'load_units': ['g'] * 4,
        'season': ['summer'] * 4,
    })

    expected_df = pandas.DataFrame({
        'A':    np.array([100., 60.]),
        'A_lo': np.array([ 90., 50.]),
        'A_hi': np.array([110., 70.]),
        'B':    np.array([ 60., 70.]),
        'B_lo': np.array([ 50., 60.]),
        'B_hi': np.array([ 70., 80.]),
        'site': ['A', 'B'],
        'parameter': ['A'] * 2,
        'load_units': ['g'] * 2,
        'season': ['summer'] * 2,
        'load_red': [40., -10/60.],
        'load_red_lower': [20./90., -30./50.],
        'load_red_upper': [60./110., 10./70.],
    })

    result_df = summary.load_reduction_pct(input_df, groupby_col='season',
                                            load_inflow='A',
                                            load_outflow='B',
                                            load_inflow_lower='A_lo',
                                            load_inflow_upper='A_hi',
                                            load_outflow_lower='B_lo',
                                            load_outflow_upper='B_hi')


def test_storm_stats():
    input_df = csv_string_as_df("""\
    site,storm_number,year,season,grouped_season,antecedent_days,start_date,end_date,duration_hours,peak_precip_intensity,total_precip_depth,runoff_m3
    ED-1,2,2011,summer,summer/autumn,3.1041,2011-07-28 08:30,2011-07-28 10:30,2.000,6.0,5.0000,16.4213
    ED-1,3,2011,summer,summer/autumn,0.6458,2011-07-29 02:00,2011-07-29 12:20,10.333,3.6,2.2,7.22535
    ED-1,5,2011,summer,summer/autumn,2.6736,2011-08-03 03:20,2011-08-03 05:20,2.000,12.0,12.8,42.0384
    ED-1,8,2011,summer,summer/autumn,2.7361,2011-08-07 04:10,2011-08-08 00:00,19.833,38.4,26.6,87.36105
    ED-1,9,2011,summer,summer/autumn,1.2638,2011-08-09 06:20,2011-08-09 14:30,8.166,20.4,34.6,113.63505
    ED-1,10,2011,summer,summer/autumn,0.3472,2011-08-09 22:50,2011-08-09 23:40,0.833,14.4,5.800,19.04865
    ED-1,14,2011,summer,summer/autumn,0.4027,2011-08-21 13:50,2011-08-21 16:40,2.833,28.8,6.0,19.7055
    ED-1,17,2011,summer,summer/autumn,0.4166,2011-08-24 20:20,2011-08-24 22:30,2.166,27.6,11.2,36.7836
    ED-1,19,2011,summer,summer/autumn,6.7430,2011-09-01 01:10,2011-09-01 02:30,1.333,4.8,5.0,16.42125
    ED-1,21,2011,summer,summer/autumn,1.8402,2011-09-03 09:20,2011-09-03 14:50,5.500,14.4,4.4,14.4507
    ED-1,24,2011,summer,summer/autumn,0.2638,2011-09-04 19:00,2011-09-04 19:50,0.833,10.8,4.0,13.137
    """, parse_dates=['start_date', 'end_date'])

    expected_df = csv_string_as_df("""\
    site,season,quantity,count,mean,std,min,25%,50%,75%,max
    ED-1,summer,antecedent_days,4.0,1.7726,1.1310,0.4167,1.0521,1.9687,2.689225,2.7361
    ED-1,summer,duration_hours,4.0,8.0417,8.3681,2.0,2.125,5.16667,11.08275,19.8333
    ED-1,summer,peak_precip_intensity,4.0,24.6,11.19285,12.0,18.3,24.0,30.3,38.4
    ED-1,summer,runoff_m3,4.0,69.9545,36.9260,36.7836,40.7247,64.6997,93.92955,113.635
    ED-1,summer,total_precip_depth,4.0,21.3,11.24334,11.2,12.4,19.7,28.6,34.6
    ED-1,summer,year,4.0,2011.0,0.0,2011.0,2011.0,2011.0,2011.0,2011.0
    """)
    expected_df.columns.names = ['stat']

    result_df = summary.storm_stats(input_df, minprecip=8,
                                    excluded_dates=[date(2011, 8, 21)],
                                    groupby_col='season')

    pdtest.assert_frame_equal(result_df, expected_df, check_less_precise=True)


def test_wq_summary():
    datecols = ['samplestart', 'samplestop', 'start_date', 'end_date']
    input_df = load_test_data('test_wq.csv', parse_dates=datecols)

    expected_none = load_test_data('wq_summary_by_nothing.csv')
    expected_season = load_test_data('wq_summary_by_season.csv')

    pdtest.assert_frame_equal(
        summary.wq_summary(input_df),
        expected_none
    )

    pdtest.assert_frame_equal(
        summary.wq_summary(input_df, groupby_col='season'),
        expected_season
    )


def test_load_totals():
    datecols = ['samplestart', 'samplestop', 'start_date', 'end_date']
    input_df = load_test_data('test_wq.csv', parse_dates=datecols)

    expected_none = load_test_data('load_totals_by_nothing.csv')
    expected_season = load_test_data('load_totals_by_season.csv')

    pdtest.assert_frame_equal(
        summary.load_totals(input_df),
        expected_none
    )

    pdtest.assert_frame_equal(
        summary.load_totals(input_df, groupby_col='season'),
        expected_season
    )
