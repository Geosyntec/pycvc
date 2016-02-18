from io import StringIO
from textwrap import dedent
from datetime import date

import nose.tools as nt
import numpy as np
import numpy.testing as nptest
import pandas
import pandas.util.testing as pdtest

from  pycvc import summary


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
    input_df = pandas.read_csv(StringIO(dedent("""\
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
    """)))

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
