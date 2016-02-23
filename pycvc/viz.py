import os
import pdb

import numpy as np
import matplotlib.pyplot as plt
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
    as[PNG|PDF] : bool, optional (default = True)
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
    """ Draws the percent reduction bar plots with the upper and lower
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
    lower = load_cols.pop('lower', 'load_reduction_lower')
    reduction = load_cols.pop('reduction', 'load_reduction')
    upper = load_cols.pop('upper', 'load_reduction_upper')
    fg = seaborn.factorplot(
        x=xvaluecol, y=reduction, aspect=1.6, size=2, hue=sitecol,
        col=paramcol, col_order=params, col_wrap=2,
        ci=None,  kind='bar', data=df,
        margin_titles=True, legend=False
    )

    fg.add_legend(ncol=2)
    fg._legend.set_bbox_to_anchor(leg_loc)
    fg.map_dataframe(_reduction_range_bars, reduction, lower, upper, sitecol)

    return fg


def hydro_pairplot(hydro, site, sitecol='site', by='season', palette=None, save=True):
    """ Creates a pairplot of hydrologic quantities.

    Parameters
    ----------
    by : string, optional (default = 'season')
        The column in Site.storm_info that defines how the data
        should be grouped.
    palette : seaborn.color_palette or None (default)
        Color scheme for the plot.

    Returns
    -------
    None

    See Also
    --------
    http://web.stanford.edu/~mwaskom/software/seaborn/generated/seaborn.pairplot.html

    """

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

    var_cols = [
        'antecedent days',
        'duration hours',
        'peak precip intensity',
        'total precip depth',
        'outflow mm',
    ]
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
        _savefig(pg.fig, figname, extra='HydroPairPlot')

    return pg


def hydro_histogram(hydro, valuecol='total_precip_depth', bins=None,
                    save=True, **factoropts):
    """ Plot a faceted, categorical histogram of storms.

    valuecol : str, optional
        The name of the column that should be categorized and plotted.
    bins : array-like, optional
        The right-edges of the histogram bins.
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

    fig = utils.figutils.categorical_histogram(
        hydro, valuecol, bins, **factoropts
    )

    if save:
        figname = 'HydroHistogram_{}'.format(valuecol)
        _savefig(fig, figname, extra='HydroHistogram')

    return fig


def hydro_jointplot(hydro, site, xcol, ycol, sitecol='site',
                    color=None, conditions=None, one2one=True,
                    save=True):
    """ Creates a joint distribution plot of two hydrologic
    quantities.

    Parameters
    ----------
    xcol, ycol : string
        Column names found in Site.storm_info
    conditions : string or None (default)
        Query strings to be passed to Site.storm_info.query(...)
    one2one : bool, optional (default = True)
        Shows the 1:1 line on the scatter portion of the joint
        distribution plot.

    Returns
    -------
    None

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
        figname = '{}-HydroJoinPlot_{}_vs_{}'.format(site, xcol, ycol)
        _savefig(jg.fig, figname, extra='HydroJointPlot')
