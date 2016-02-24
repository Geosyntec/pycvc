import os
import csv
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import pandas
import seaborn.apionly as seaborn

import wqio
from wqio import utils

from . import dataAccess
from . import info
from . import viz
from .external import bmpcats_to_use
from . import validate


def collect_tidy_data(sites, fxn):
    return pandas.concat([fxn(site) for site in sites], ignore_index=True)


def labels_from_bins(bins, units=None):
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if left == 0:
            labels.append('<{}'.format(right))
        elif np.isinf(right):
            labels.append('>{}'.format(left))
        else:
            labels.append('{} - {}'.format(left, right))

    if units is not None:
        labels = map(lambda c: '{} {}'.format(c, units), labels)

    return list(labels)


def classify_storms(df, valuecol, newcol='storm_bin', bins=None):
    if bins is None:
        bins = [0, 5, 10, 15, 20, 25, np.inf]

    labels = labels_from_bins(bins=bins, units='mm')
    df[newcol] = pandas.cut(df[valuecol], bins=bins, labels=labels)
    return df


def prevalence_table(wq, rescol='concentration', groupby_col=None):
    """ Returns a sample prevalence table for the given sample type.
    """
    by = ['site', 'parameter']
    if groupby_col is not None:
        by.append(validate.groupby_col(groupby_col))

    pt = (
        wq.query("sampletype == 'composite'")
            .groupby(by=by)
            .count()[rescol]
            .unstack(level='parameter')
            .reset_index()
    )
    return pt


def remove_load_data_from_storms(wq, stormdates, datecol):
    if np.isscalar(stormdates):
        stormdates = [stormdates]

    cols_to_clean = wq.select(lambda c: c.startswith('load_'), axis=1).columns
    row_to_clean = wq[datecol].dt.date.isin(stormdates)
    wq = wq.copy()
    wq.loc[row_to_clean, cols_to_clean] = np.nan
    return wq


def pct_reduction(wq, incol, outcol):
    return 100 * (wq[incol] - wq[outcol]) / wq[incol]


def load_reduction_pct(wq, groupby_col=None, **load_cols):
    load_in = load_cols.pop('load_inflow', 'load_inflow')
    load_out = load_cols.pop('load_outflow', 'load_outflow')
    load_in_lower = load_cols.pop('load_inflow_lower', 'load_inflow_lower')
    load_in_upper = load_cols.pop('load_inflow_upper', 'load_inflow_upper')
    load_out_lower = load_cols.pop('load_outflow_lower', 'load_outflow_lower')
    load_out_upper = load_cols.pop('load_outflow_upper', 'load_outflow_upper')

    by = ['site', 'parameter', 'load_units']
    if groupby_col is not None:
        by.append(groupby_col)

    red = (
        wq.groupby(by=by)
            .sum()
            .assign(load_red=lambda df: pct_reduction(df, load_in, load_out))
            .assign(load_red_lower=lambda df: pct_reduction(df, load_in_lower, load_out_upper))
            .assign(load_red_upper=lambda df: pct_reduction(df, load_in_upper, load_out_lower))
            .select(lambda c: c.startswith('load_red'), axis=1)
            .join(wq.groupby(by=by).size().to_frame())
            .rename(columns={0: 'Count'})
            .dropna()
            .reset_index()
    )

    return red


@np.deprecate
def write_load_reduction_range(reduction_df, site):  # pragma: no cover
    final_cols = [
        'Total Suspended Solids',
        'Cadmium (Cd)',
        'Copper (Cu)',
        'Iron (Fe)',
        'Lead (Pb)',
        'Nickel (Ni)',
        'Zinc (Zn)',
        'Nitrate (N)',
        'Orthophosphate (P)',
        'Total Kjeldahl Nitrogen (TKN)',
        'Total Phosphorus',
    ]
    reduction_df = (
        reduction_df.applymap(lambda x: utils.sigFigs(x, n=2))
        .apply(lambda r: '{} - {}'.format(r['load_red_lower'], r['load_red_upper']), axis=1)
        .unstack(level='parameter')
    )[final_cols]

    reduction_df.xs(site, level='site').to_csv('{}_reduction.csv'.format(site), quoting=csv.QUOTE_ALL)


@np.deprecate
def load_summary_table(wq):  # pragma: no cover
    """ Produces a summary table of loads and confidence intervals for
    varying sample/event types (e.g., composite, unsampled w/ or w/o
    outflor) from tidy water quality data.
    """
    def set_column_name(df):
        df.columns.names = ['quantity']
        return df

    def formatter(row, location):
        cols = 'load_{0};load_{0}_lower;load_{0}_upper'.format(location).split(';')
        row_str = "{} ({}; {})".format(*row[cols])
        return row_str

    def drop_unit_index(df):
        df = df.copy()
        df.index = df.index.droplevel('units')
        return df

    def set_no_outlow_zero(df):
        col = ('Effluent', 'unsampled', 'No')
        df[col] = 0
        return df

    def set_measured_effl(df):
        col = ('Effluent', 'composite', 'Yes')
        df[col] = df[col].apply(lambda x: str(x).split(' ')[0])
        return df

    def swal_col_levels(df, ii, jj):
        df.columns = df.columns.swaplevel(ii, jj)
        return df

    def pct_reduction(df, incol, outcol):
        return 100 * (df[incol] - df[outcol]) / df[incol]


    final_cols = [
        ('Influent', 'unsampled', 'No'), ('Effluent', 'unsampled', 'No'),
        ('Influent', 'unsampled', 'Yes'), ('Effluent', 'unsampled', 'Yes'),
        ('Influent', 'composite', 'Yes'), ('Effluent', 'composite', 'Yes'),
        'Reduction'
    ]

    final_params = [
        'Nitrate + Nitrite', 'Nitrate (N)', 'Orthophosphate (P)',
        'Cadmium (Cd)', 'Copper (Cu)', 'Iron (Fe)',
        'Total Kjeldahl Nitrogen (TKN)', 'Lead (Pb)',
        'Nickel (Ni)', 'Total Phosphorus',
        'Total Suspended Solids', 'Zinc (Zn)',
    ]

    loads = (
        wq.groupby(by=['parameter', 'units', 'has_outflow', 'sampletype'])
          .sum()
          .pipe(set_column_name)
          .select(lambda c: c.startswith('load_inflow') or c.startswith('load_outflow'), axis=1)
          .unstack(level='has_outflow')
          .pipe(swal_col_levels, 'has_outflow', 'quantity')
          .stack(level='has_outflow')
          .dropna()
          .pipe(drop_unit_index)
    )

    main = (
        loads.applymap(lambda x: utils.sigFigs(x, n=3, expthresh=7))
          .assign(Influent=lambda df: df.apply(formatter, args=('inflow',), axis=1))
          .assign(Effluent=lambda df: df.apply(formatter, args=('outflow',), axis=1))
          .unstack(level='sampletype')
          .unstack(level='has_outflow')
          [['Influent', 'Effluent']]
          .dropna(how='all', axis=1)
    )

    reduction = (
        loads.groupby(level='parameter').sum()
             .assign(load_red=lambda df: pct_reduction(df, 'load_inflow', 'load_outflow'))
             .assign(load_red_upper=lambda df: pct_reduction(df, 'load_inflow_upper', 'load_outflow_lower'))
             .assign(load_red_lower=lambda df: pct_reduction(df, 'load_inflow_lower', 'load_outflow_upper'))
             .applymap(lambda x: utils.sigFigs(x, n=3, expthresh=7))
             .assign(Reduction=lambda df: df.apply(formatter, args=('red',), axis=1))
    )

    summary = (
        main.join(reduction)
            .loc[final_params, final_cols]
            .pipe(set_no_outlow_zero)
            .pipe(set_measured_effl)
    )

    return summary


def storm_stats(hydro, minprecip=0, excluded_dates=None, groupby_col=None):
    """ Statistics summarizing all the storm events

    Parameters
    ----------
    minprecip : float (default = 0)
        The minimum amount of precipitation required to for a storm
        to be included. Using 0 (the default) will likely include
        some pure snowmelt events.
    excluded_dates : list of date-likes, optional
        This is a list of storm start dates that will be removed
        from the storms dataframe prior to computing statistics.
    groupby_col : string, optional
        Optional string that defined how results should be group
        temporally. Valid options are "season", "grouped_season",
        and year. Default behavior does no temporal grouping.
    **winsor_params : optional keyword arguments
        Dictionary of column names (from `Site.storm_info`) and
        percetiles at which those columns should be winsorized.

    Returns
    -------
    summary : pandas.DataFrame

    See also
    --------
    pycvc.Site.storm_info
    wqio.units.winsorize_dataframe
    scipy.stats.mstats.winsorize

    """

    timecol = validate.groupby_col(groupby_col)

    by = ['site']
    if groupby_col is not None:
        by.append(validate.groupby_col(groupby_col))

    data = (
        hydro.pipe(dataAccess._remove_storms_from_df, excluded_dates, 'start_date')
            .query("total_precip_depth > @minprecip")
    )

    descr = data.groupby(by=by).describe()
    descr.index.names = by + ['stat']
    descr = descr.select(lambda c: c != 'storm_number', axis=1)
    descr.columns.names = ['quantity']
    storm_stats = (
        descr.stack(level='quantity')
             .unstack(level='stat')
             .reset_index()
    )
    return storm_stats


def wq_summary(wq, rescol='concentration', sampletype='composite',
               groupby_col=None):

    """ Basic water quality Statistics

    Parameters
    ----------
    rescol : string (default = 'concentration')
        The result column to summaryize. Valid values are
        "concentration" and "load_outflow".
    sampletype : string (default = 'composite')
        The types of samples to be summarized. Valid values are
        "composite" and "grab".
    groupby_col : string, optional
        Optional string that defined how results should be group
        temporally. Valid options are "season", "grouped_season",
        and year. Default behavior does no temporal grouping.

    Returns
    -------
    summary : pandas.DataFrame

    """

    rescol, unitscol = validate.rescol(rescol)
    sampletype = validate.sampletype(sampletype)

    by = ['site', 'parameter', unitscol]
    if groupby_col is not None:
        by.append(validate.groupby_col(groupby_col))

    summary_percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # detects and non-detects
    all_data = (
        wq.query("sampletype == @sampletype")
            .groupby(by=by)[rescol]
            .apply(lambda g: g.describe(percentiles=summary_percentiles))
            .unstack(level=-1)
    )

    # count non-detects
    if wq.query("sampletype == @sampletype and qualifier != '='").shape[0] > 0:
        nd_data = (
            wq.query("sampletype == @sampletype and qualifier != '='")
                .groupby(by=by)[rescol]
                .size()
                .to_frame()
                .rename(columns={0: 'count NDs'})
        )

        all_data = all_data.join(nd_data).fillna({'count NDs': 0})
    else:
        all_data['count NDs'] = 0

    # compute the coefficient of variation
    all_data['cov'] = all_data['std'] / all_data['mean']

    # fancy column order
    columns = [
        'count', 'count NDs', 'mean', 'std', 'cov',
        'min', '10%', '25%', '50%', '75%', '90%', 'max',
    ]

    # columns to rename
    stat_labels = {
        'count': 'Count',
        'count NDs': 'Count of Non-detects',
        'mean': 'Mean',
        'std': 'Standard Deviation',
        'cov': 'Coeff. of Variation',
        'min': 'Minimum',
        '10%': '10th Percentile',
        '25%': '25th Percentile',
        '50%': 'Median',
        '75%': '75th Percentile',
        '90%': '90th Percentile',
        'max': 'Maximum',
    }

    return all_data[columns].rename(columns=stat_labels).reset_index()


def load_totals(wq, groupby_col=None, NAval=0):
    """ Returns the total loads for sampled storms and the given
    sampletype.

    Parameters
    ----------
    sampletype : string (default = 'composite')
        The types of samples to be summarized. Valid values are
        "composite" and "grab".
    groupby_col : string, optional
        Optional string that defined how results should be group
        temporally. Valid options are "season", "grouped_season",
        and year. Default behavior does no temporal grouping.
    excluded_dates : list of date-likes, optional
        This is a list of storm start dates that will be removed
        from the storms dataframe prior to computing statistics.
    NAval : float, optional
        Default value with which NA (missing) loads will be filled.
        If none, NAs will remain inplace.

    Returns
    -------
    sampled_loads : pandas.DataFrame

    """

    by = ['site', 'parameter', 'sampletype', 'has_outflow', 'load_units']
    if groupby_col is not None:
        by.append(validate.groupby_col(groupby_col))

    agg_dict = {
        'units': 'first',
        'load_units': 'first',
        'load_runoff_lower': 'sum',
        'load_runoff': 'sum',
        'load_runoff_upper': 'sum',
        'load_inflow_lower': 'sum',
        'load_inflow': 'sum',
        'load_inflow_upper': 'sum',
        'load_bypass_lower': 'sum',
        'load_bypass': 'sum',
        'load_bypass_upper': 'sum',
        'load_outflow_lower': 'sum',
        'load_outflow': 'sum',
        'load_outflow_upper': 'sum',
    }

    def total_reduction(df, incol, outcol):
        return df[incol] - df[outcol]

    def pct_reduction(df, redcol, incol):
        return df[redcol] / df[incol] * 100.0

    final_cols_order = [
        'load_runoff_lower',
        'load_runoff',
        'load_runoff_upper',
        'load_bypass_lower',
        'load_bypass',
        'load_bypass_upper',
        'load_inflow_lower',
        'load_inflow',
        'load_inflow_upper',
        'load_outflow_lower',
        'load_outflow',
        'load_outflow_upper',
        'reduct_mass_lower',
        'reduct_mass',
        'reduct_mass_upper',
        'reduct_pct_lower',
        'reduct_pct',
        'reduct_pct_upper',
    ]

    final_cols = [
        'Runoff Load (lower bound)',
        'Runoff Load',
        'Runoff Load (upper bound',
        'Bypass Load (lower bound)',
        'Bypass Load',
        'Bypass Load (upper bound',
        'Estimated Total Influent Load (lower bound)',
        'Estimated Total Influent Load',
        'Estimated Total Influent Load (upper bound',
        'Total Effluent Load (lower bound)',
        'Total Effluent Load',
        'Total Effluent Load (upper bound',
        'Load Reduction Mass (lower bound)',
        'Load Reduction Mass',
        'Load Reduction Mass (upper bound)',
        'Load Reduction Percent (lower bound)',
        'Load Reduction Percent',
        'Load Reduction Percent (upper bound)',
    ]

    loads = (
        wq.groupby(by=by)
            .agg(agg_dict)
            .fillna(NAval)
            .assign(reduct_mass_lower=lambda df: total_reduction(df, 'load_inflow_lower', 'load_outflow_upper'))
            .assign(reduct_pct_lower=lambda df: pct_reduction(df, 'reduct_mass_lower', 'load_inflow_lower'))
            .assign(reduct_mass=lambda df: total_reduction(df, 'load_inflow', 'load_outflow'))
            .assign(reduct_pct=lambda df: pct_reduction(df, 'reduct_mass', 'load_inflow'))
            .assign(reduct_mass_upper=lambda df: total_reduction(df, 'load_inflow_upper', 'load_outflow_lower'))
            .assign(reduct_pct_upper=lambda df: pct_reduction(df, 'reduct_mass_upper', 'load_inflow_upper'))
            .rename(columns=dict(zip(final_cols_order, final_cols)))
    )[final_cols]

    return loads.reset_index()
