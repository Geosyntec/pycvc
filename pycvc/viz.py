import os
import pdb

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn.apionly as seaborn

from wqio import utils


def savefig(fig, figname, extra=None, asPNG=True, asPDF=False, load=False):
    """
    Helper function to effectively save a figure.

    Parameters
    ----------
    fig : matplotlib.Figure
        The figure to be written to disk.
    figname : string
        The name of the image of the figure without any file extension.
    extra : string or None (default)
        Relative path to subdirectories below './output/img' where the
        figure will be saved.
    asPNG, asPDF : bool, optional (defaults: True, False, respectively)
        Toggles if the figure will be saved as a .png and/or .pdf file.
        If both are False, the figure is not saved in any form.
    load : bool, optional (default = False)
        If True, "-load" is appended to ``figname``.

    Writes
    ------
    Images of the matplotlib figure.

    Returns
    -------
    None

    """

    if load:
        figname += '-load'

    figpath = os.path.join('output', 'img')
    if extra is not None:
        figpath = os.path.join(figpath, extra)

    figpathname = os.path.join(figpath, figname)
    figopts = dict(dpi=300, transparent=True, bbox_inches='tight')

    if asPNG:
        fig.savefig(figpathname + '.png', **figopts)

    if asPDF:
        fig.savefig(figpathname + '.pdf', **figopts)

    plt.close(fig)


def _reduction_range_bars(y, ylow, yhigh, sitecol, data=None, **kwargs):
    """ Draws upper/lower bound bars on `reduction_plots` """
    ax = plt.gca()
    offset_map = {
        'ED-1': 0.0,
        'LV-2': -0.2,
        'LV-4': +0.2
    }
    data = (
        data.assign(offset=data[sitecol].map(offset_map))
            .assign(err_low=data[y] - data[ylow])
            .assign(err_high=data[yhigh] - data[y])
    )

    value_selection = None if data[sitecol].unique().shape[0] > 1 else 0
    data['x'] = (
        data.groupby(sitecol)
            .apply(lambda g: np.arange(g['offset'].size) + g['offset'])
            .values
            .flatten()
    )
    ax.errorbar(x=data['x'], y=data[y], yerr=data[['err_low', 'err_high']].values.T, ls='none', fmt='k')


def reduction_plot(df, params, paramcol, sitecol, xvaluecol, leg_loc, **load_cols):
    """
    Draws the percent reduction bar plots with the upper and lower
    estimates.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with columns for the parameter, site, x-values, and
        load reduction estimates (lower, primary and upper estimates).
    params : list of str
        The names of the parameters to be included in the plot.
    paramcol, sitecol, xvaluecol : str
        Column labels in `df` that represent the columns containing the
        parameters, site labels, and x-axis value to be shown in the
        plots.
    leg_loc : tuple of floats
        The legend for these figures is built manually to the figure.
        `leg_loc` is a tuple of `(xpos, ypos)` of the lower left corner
        of the legend relative to the lower left corner of the figure.
        In other words `(0.0, 0.0)` will place the legend firmly in the
        lower left corner of the figure, while `(1.0, 1.0)` will place
        the legend just beyon the upper right corner.
    lower, reduction, upper : string, optional
        Column labels for the percent reduction estimates. Default
        values are 'load_reduction_lower', 'load_reduction', and
        'load_reduction_upper', respectively.

    Returns
    -------
    fg : seaborn FacetGrid
        The FaceGrid on which the data have been drawn.

    """

    # pull out columns from the kwargs
    lower = load_cols.pop('lower', 'load_reduction_lower')
    reduction = load_cols.pop('reduction', 'load_reduction')
    upper = load_cols.pop('upper', 'load_reduction_upper')

    # draw the plot
    fg = seaborn.factorplot(
        x=xvaluecol, y=reduction, aspect=1.6, size=2, hue=sitecol,
        col=paramcol, col_order=params, col_wrap=2,
        ci=None,  kind='bar', data=df,
        margin_titles=True, legend=False
    )

    # add a legend
    fg.add_legend(ncol=2)
    fg._legend.set_bbox_to_anchor(leg_loc)

    # draw the error bars
    fg.map_dataframe(_reduction_range_bars, reduction, lower, upper, sitecol)

    return fg


def hydro_pairplot(hydro, site, sitecol='site', by='season',
                   palette=None, save=True):
    """
    Creates a pairplot of hydrologic quantities.

    Parameters
    ----------
    hydro : pandas.DataFrame
        A tidy dataframe of the hydrologic info of storms.
    site : string
        The ID string of the site to be plotted (e.g., 'ED-1' for Elm
        Drive)
    sitecol : string
        The label of the column in ``hydro`` that contains ``site``.
    by : string, optional (default = 'season')
        The column in Site.storm_info that defines how the data
        should be grouped.
    palette : seaborn.color_palette, optional
        Color scheme for the plot. Defaults to the current palette.
    save : bool, optional (False)
        If True, the figure is automatically saved via :func:savefig
        in the default locations ('output/img/HydroPairPlot')

    Returns
    -------
    pg : seaborn.PairGrid

    See Also
    --------
    http://web.stanford.edu/~mwaskom/software/seaborn/generated/seaborn.pairplot.html

    """

    # full list of columns for the dataframe
    cols = [
        'site',
        'antecedent_days',
        'duration_hours',
        'peak_precip_intensity',
        'total_precip_depth',
        'outflow_mm',
        'season',
        'Seasons',
        'year',
        'Has outflow?'
    ]

    # just the numeric columns to be plotted
    var_cols = [
        'antecedent days',
        'duration hours',
        'peak precip intensity',
        'total precip depth',
        'outflow mm',
    ]

    # clean up column names, drop irrelevant columns
    sinfo = (
        hydro[hydro[sitecol] == site]
            .rename(columns={'has_outflow': 'Has outflow?', 'grouped_season': 'Seasons'})
            .select(lambda c: c in cols, axis=1)
            .rename(columns=lambda c: c.replace('_', ' '))
    )

    if by == 'season':
        opts = dict(
            palette=palette or 'BrBG_r',
            hue='season',
            markers=['o', 's', '^', 'd'],
            hue_order=['winter', 'spring', 'summer', 'autumn'],
            vars=var_cols
        )

    elif by =='year':
        opts = dict(
            hue='year',
            palette=palette or 'deep',
            vars=var_cols
        )

    elif by == 'outflow':
        opts = dict(
            hue='Has outflow?',
            palette=palette or 'deep',
            markers=['o', 's'],
            hue_order=['Yes', 'No'],
            vars=var_cols
        )

    elif by == 'grouped_season':
        opts = dict(
            palette=palette or 'BrBG_r',
            hue='Seasons',
            markers=['o', '^'],
            hue_order=['winter/spring', 'summer/autumn'],
            vars=var_cols
        )

    pg = seaborn.pairplot(sinfo, **opts)
    for ax in pg.axes.flat:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    if save:
        figname = '{}-HydroPairPlot_by_{}'.format(site, by)
        savefig(pg.fig, figname, extra='HydroPairPlot')

    return pg


def hydro_histogram(hydro, valuecol='total_precip_depth', bins=None,
                    save=True, **factoropts):
    """
    Plot a faceted, categorical histogram of storms.

    Parameters
    ----------
    hydro : pandas.DataFrame
        A tidy dataframe of the hydrologic info of storms.
    valuecol : str, optional ('total_precip_depth')
        The name of the column that should be categorized and plotted.
    bins : array-like, optional
        The edges of the histogram bins.
    save : bool, optional (False)
        If True, the figure is automatically saved via :func:savefig
        in the default locations ('output/img/HydroHistogram')
    factoropts : keyword arguments, optional
        Options passed directly to seaborn.factorplot

    Returns
    -------
    fig : seaborn.FacetGrid

    See also
    --------
    utils.figutils.categorical_histogram
    seaborn.factorplot

    """

    if bins is None:
        bins = np.arange(5, 30, 5)

    fg = utils.figutils.categorical_histogram(
        hydro, valuecol, bins, **factoropts
    )

    if save:
        figname = 'HydroHistogram_{}'.format(valuecol)
        savefig(fg.fig, figname, extra='HydroHistogram')

    return fg


def hydro_jointplot(hydro, site, xcol, ycol, sitecol='site',
                    color=None, conditions=None, one2one=True,
                    save=True):
    """
    Creates a joint distribution plot of two hydrologic quantities.

    Parameters
    ----------
    hydro : pandas.DataFrame
        A tidy dataframe of the hydrologic info of storms.
    site : string
        The ID string of the site to be plotted (e.g., 'ED-1' for Elm
        Drive)
    sitecol, xcol, ycol : string
        Column names found in ``hydro`` that specify the columns
        containing the site IDs, x-values, and y-values, respectively.
    conditions : string or None (default)
        Query strings to be passed to ``hydro.query(...)``
    one2one : bool, optional (default = True)
        Shows the 1:1 line on the scatter portion of the joint
        distribution plot.
    save : bool, optional (False)
        If True, the figure is automatically saved via :func:savefig
        in the default locations ('output/img/HydroJoint')

    Returns
    -------
    jg : seaborn.JointGrid

    See Also
    --------
    http://web.stanford.edu/~mwaskom/software/seaborn/generated/seaborn.jointplot.html

    """

    column_labels = {
        'total_precip_depth': 'Storm Precipitation Depth (mm)',
        'peak_precip_intensity': 'Peak Precipitation Intensity (mm/hr)',
        'outflow_mm': 'BMP Outflow (watershed mm)',
        'peak_outflow': 'Peak BMP Outflow (L/s)',
        'duration_hours': 'Storm Duration (hr)',
        'antecedent_days': 'Antecedent Dry Period (days)',
    }

    data = hydro[hydro[sitecol] == site]
    if conditions is not None:
        data = data.query(conditions)

    jg = utils.figutils.jointplot(
        x=xcol, y=ycol, data=data, one2one=one2one, color=color,
        xlabel=column_labels[xcol], ylabel=column_labels[ycol],
    )

    if save:
        figname = '{}-HydroJointPlot_{}_vs_{}'.format(site, xcol, ycol)
        savefig(jg.fig, figname, extra='HydroJointPlot')


def external_boxplot(combined, sites=None, categories=None, params=None,
                     units=None, palette=None):
    """
    Faceted box and whisker plots of site data with external (i.e.,
    NSQD or BMPDB data). Since we're comparing CVC with external data,
    only concentrations can be plotted (i.e., loads from BMPDB and NSQD
    are not available).

    Parameters
    ----------
    combined : pandas.DataFrame
        A single tidy dataframe of both the CVC and external data.

        .. note ::
           BMP categories or NSQD landuses should be stored in the
           `'site'` column of the dataframe.

    sites : list of string
        The CVC sites to include in the plot.
    categories : list of string
        The NSQD landuses or BMPDB categories to include in the plot.
    params : list of string
        The parameters to include in the plots.
    units : string
        The units of measure of the quantity being plotted.
    palette : seaborn.color_palette, optional
        Color scheme for the plot. Defaults to the current palette.

    Returns
    -------
    fg : seaborn.FacetGrid

    Examples
    --------
    >>> import pandas
    >>> import pycvc
    >>> bmpdb = pycvc.external.bmpdb('black', 'D')
    >>> tidy_file = 'output/tidy/wq_simple.csv'
    >>> datecols = ['start_date', 'end_date', 'samplestart', 'samplestop']
    >>> wq = (
    ...     pandas.read_csv(tidy_file, parse_dates=datecols)
    ...         .pipe(pycvc.summary.classify_storms, 'total_precip_depth')
    ...         .pipe(pycvc.summary.remove_load_data_from_storms, ['2013-07-08'], 'start_date')
    ... )
    >>> combined = pycvc.external.combine_wq(wq, bmpdb, 'category')
    >>> pycvc.viz.external_boxplot(
    ...     tidy=combined,
    ...     sites=['ED-1', 'LV-2', 'LV-4'],
    ...     categories=['Bioretention', 'Detention Basin', 'Wetland Channel'],
    ...     palette='Blues',
    ...     params=['Cadmium (Cd)', 'Copper (Cu)', 'Lead (Pb)', 'Zinc (Zn)'],
    ...     units='μg/L'
    ... )

    """

    x_vals = np.hstack([sites, categories])

    subset = (
        combined.query("site in @x_vals")
                .query("parameter in @params")
    )
    fg = seaborn.factorplot(
        data=subset, x='site', y='concentration',
        col='parameter', col_wrap=2, col_order=params,
        kind='box', aspect=2, size=3, order=x_vals,
        palette=palette, sharey=False, notch=False
    )

    xlabels = list(map(lambda c: c.replace(' ', '\n'), x_vals))
    _format_facetgrid(fg, units, xlabels=xlabels)
    return fg


def seasonal_boxplot(wq, ycol, params, units, palette=None):
    """
    Faceted box and whisker plots of site data grouped by the season
    during which the sample was collected.

    Parameters
    ----------
    wq : pandas.DataFrame
        A single tidy dataframe of the CVC water quality data.
    ycol : string
        The label of the column you wish to plot in the boxplots (e.g.,
        `'concentration'` or `'load_outflow'`).
    params : list of string
        The parameters to include in the plots.
    units : string
        The units of measure of the quantity being plotted.
    palette : seaborn.color_palette, optional
        Color scheme for the plot. Defaults to 'BrBG_r' (minty-green for
        winter/spring to brown for summer/autumn).

    Returns
    -------
    fg : seaborn.FacetGrid

    Examples
    --------
    >>> import pycvc
    >>> tidy_file = 'output/tidy/wq_simple.csv'
    >>> wq = (
    ...     pandas.read_csv(tidy_file, parse_dates=['start_date', 'end_date'])
    ...         .pipe(pycvc.summary.classify_storms, 'total_precip_depth')
    ...         .pipe(pycvc.summary.remove_load_data_from_storms, ['2013-07-08'], 'start_date')
    ... )
    >>> params = ['Cadmium (Cd)', 'Copper (Cu)', 'Lead (Pb)', 'Zinc (Zn)']
    >>> bp = pycvc.viz.seasonal_boxplot(wq, 'concentration', params, 'μg/L')

    """

    if palette is None:
        palette = 'BrBG_r'

    fg = seaborn.factorplot(
        data=wq.query('parameter in @params'), x='site', y='concentration',
        col='parameter', col_wrap=2, col_order=params,
        hue='season', hue_order=['winter', 'spring', 'summer', 'autumn'],
        kind='box', palette=palette, aspect=2, size=3, sharey=False
    )

    _format_facetgrid(fg, units)

    return fg


def ts_plot(wq, datecol, ycol, sites, params, units,
            palette=None, markers=None):
    """
    Faceted time series plots of CVC water quality data.

    Parameters
    ----------
    wq : pandas.DataFrame
        A single tidy dataframe of the CVC water quality data.
    datecol, ycol : string
        The label of the column with the dates and the column you wish
        to plot as the y-values (e.g., `'concentration'` or
        `'load_outflow'`).
    sites : list of string
        The CVC sites to include in the plot.
    params : list of string
        The parameters to include in the plots.
    units : string
        The units of measure of the quantity being plotted.
    palette : seaborn.color_palette, optional
        Color scheme for the plot. Defaults to 'BrBG_r' (minty-green for
        winter/spring to brown for summer/autumn).
    markers : list of string, optional
        List of valid matplotlib markers. This should have the same
        number of elements as ``sites``.

    Returns
    -------
    fg : seaborn.FacetGrid

    Examples
    --------
    >>> import pycvc
    >>> tidy_file = 'output/tidy/wq_simple.csv'
    >>> wq = (
    ...     pandas.read_csv(tidy_file, parse_dates=['start_date', 'end_date'])
    ...         .pipe(pycvc.summary.classify_storms, 'total_precip_depth')
    ...         .pipe(pycvc.summary.remove_load_data_from_storms, ['2013-07-08'], 'start_date')
    ... )
    >>> params = ['Cadmium (Cd)', 'Copper (Cu)', 'Lead (Pb)', 'Zinc (Zn)']
    >>> sites = ['ED-1', 'LV-2', 'LV-4']
    >>> ts = pycvc.viz.ts_plot(wq, 'samplestart', 'concentration',
    ...                        sites, params, 'μg/L',
    ...                        markers=['o', 's', '^'])

    """

    subset = wq.query("site in @sites and parameter in @params").dropna(subset=[ycol])
    fg = seaborn.FacetGrid(
        subset, aspect=2, size=3, sharey=False,
        col_order=params, col='parameter', col_wrap=2,
        hue='site', hue_kws=dict(marker=markers),
        palette=palette,
    ).map(plt.scatter, 'samplestart', ycol).add_legend()
    fg = _format_facetgrid(fg, units)
    return fg


def prob_plot(wq, ycol, sites, params, units, palette=None, markers=None):
    """
    Faceted probability plots of CVC water quality data.

    Parameters
    ----------
    wq : pandas.DataFrame
        A single tidy dataframe of the CVC water quality data.
    ycol : string
        The label of the column you wish to plot as the y-values (e.g.,
        `'concentration'` or `'load_outflow'`).
    sites : list of string
        The CVC sites to include in the plot.
    params : list of string
        The parameters to include in the plots.
    units : string
        The units of measure of the quantity being plotted.
    palette : seaborn.color_palette, optional
        Color scheme for the plot. Defaults to 'BrBG_r' (minty-green for
        winter/spring to brown for summer/autumn).
    markers : list of string, optional
        List of valid matplotlib markers. This should have the same
        number of elements as ``sites``.

    Returns
    -------
    fg : seaborn.FacetGrid

    Examples
    --------
    >>> import pycvc
    >>> tidy_file = 'output/tidy/wq_simple.csv'
    >>> wq = (
    ...     pandas.read_csv(tidy_file, parse_dates=['start_date', 'end_date'])
    ...         .pipe(pycvc.summary.classify_storms, 'total_precip_depth')
    ...         .pipe(pycvc.summary.remove_load_data_from_storms, ['2013-07-08'], 'start_date')
    ... )
    >>> params = ['Cadmium (Cd)', 'Copper (Cu)', 'Lead (Pb)', 'Zinc (Zn)']
    >>> sites = ['ED-1', 'LV-2', 'LV-4']
    >>> pp = pycvc.viz.prob_plot(wq, concentration', sites, params,
    ...                          'μg/L', markers=['o', 's', '^'])

    """

    def _pp(x, **kwargs):
        ax = plt.gca()
        qntls, xr = stats.probplot(x, fit=False)
        probs = stats.norm.cdf(qntls) * 100
        ax.scatter(probs, xr, **kwargs)
        ax.set_xlim(left=0.5, right=99.5)
        ax.set_xscale('prob')

    subset = wq.query("site in @sites and parameter in @params").dropna(subset=[ycol])
    fg = seaborn.FacetGrid(
        subset, aspect=2, size=3, sharey=False,
        col_order=params, col='parameter', col_wrap=2,
        hue='site', hue_kws=dict(marker=markers),
        palette=palette,
    ).map(_pp, ycol).add_legend()
    fg = _format_facetgrid(fg, units)
    fg.set_xlabels('Probability')
    return fg


def _format_facetgrid(fg, units, yval='Concentration', xlabels=None):
    """
    Rotates the ticklabels, sets the yscale='log', etc
    """
    fg.set_xticklabels(
        labels=xlabels,
        rotation=30,
        rotation_mode='anchor',
        horizontalalignment='right'
    )
    fg.set(yscale='log')
    fg.set_xlabels('')
    fg.set_axis_labels(x_var='', y_var='{} ({})'.format(yval, units))
    return fg
