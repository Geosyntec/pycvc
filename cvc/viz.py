import os
import pdb

import matplotlib.pyplot as plt
import seaborn.apionly as seaborn


def _savefig(fig, figname, extra=None, asPNG=True, asPDF=False, load=False):
    """ Helper function to effectively save a figure.

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


def _try_to_plot_ts(data, rescol, datecol, **plotkwds):
    """ Helper function to check that a dataframe is not empty and then
    to plot the data as a time series

    Parameters
    ----------
    data :pandas.DataFrame
        The data that you want to plot
    rescol, datecol : str
        Labels of the columns that contrain the result (y) and date (x)
        values of the plot.
    plotkwds : keyword arguments
        Valid matplotlib/pandas kwargs for plotting passed directly to
        ``pandas.DataFrame.plot()``

    Writes
    ------
    None

    Returns
    -------
    N : int
        The number of records in the dataframe.

    """

    label = plotkwds.pop('label', None)
    N = data.shape[0]
    datatoplot = data[[datecol, rescol]].set_index(datecol)
    if label is not None:
        datatoplot = datatoplot.rename(columns={rescol: label})

    if N > 0:
        datatoplot.plot(legend=False, **plotkwds)

    return N


def _make_time_series(wqcomp, axes, addNote, load=False):
    """ Time series helper function

    Parameters
    ----------
    wqcomp : cvc.summary.WQComparison object
        The comparison object to be plotted
    axes : matplotlib.Axes
        The Axes object on which the data should be plotted.
    addNote : bool
        Whether or not the note should be added.
    load : bool
        Whether or not the data represents loads. Otherwise
        concentration data are assumed.
    """

    wqcomp.parameterTimeSeries(ax=axes[0], addNote=addNote, addLegend=False,
                               finalOutput=False, load=load)


def _make_statplots(wqcomp, axes, addNote, load=False):
    """ Statplot helper function

    Parameters
    ----------
    wqcomp : cvc.summary.WQComparison object
        The comparison object to be plotted
    axes : matplotlib.Axes
        The Axes object on which the data should be plotted.
    addNote : bool
        Whether or not the note should be added.
    load : bool
        Whether or not the data represents loads. Otherwise
        concentration data are assumed.
    """

    wqcomp.parameterStatPlot(ax1=axes[0], ax2=axes[1], finalOutput=False,
                             labelax1=False, load=load)


def _make_bmp_boxplot(wqcomp, axes, addNote, load=False):
    """ BMP database comparison boxplot helper function

    Parameters
    ----------
    wqcomp : cvc.summary.WQComparison object
        The comparison object to be plotted
    axes : matplotlib.Axes
        The Axes object on which the data should be plotted.
    addNote : bool
        Whether or not the note should be added.
    load : bool
        Whether or not the data represents loads. Otherwise
        concentration data are assumed.
    """

    wqcomp.bmpCategoryBoxplots(ax=axes[0], addNote=addNote, finalOutput=False)


def _make_landuse_boxplot(wqcomp, axes, addNote, load=False):
    """ NSQD comparison boxplot helper function

    Parameters
    ----------
    wqcomp : cvc.summary.WQComparison object
        The comparison object to be plotted
    axes : matplotlib.Axes
        The Axes object on which the data should be plotted.
    addNote : bool
        Whether or not the note should be added.
    load : bool
        Whether or not the data represents loads. Otherwise
        concentration data are assumed.
    """

    wqcomp.landuseBoxplots(ax=axes[0], addNote=addNote, finalOutput=False)


def _make_seasonal_boxplot(wqcomp, axes, addNote=False, load=False):
    """ Seasonal comparison boxplot helper function

    Parameters
    ----------
    wqcomp : cvc.summary.WQComparison object
        The comparison object to be plotted
    axes : matplotlib.Axes
        The Axes object on which the data should be plotted.
    addNote : bool
        Whether or not the note should be added.
    load : bool
        Whether or not the data represents loads. Otherwise
        concentration data are assumed.
    """

    wqcomp.seasonalBoxplots(ax=axes[0], load=load)


def formatGSAxes(ax, axtype, col, xticks, ylabel, sublabel=None, labelsize=8):
    """ Helper function to format the Gridspec Axes in a MegaFigure

    Parameters
    ----------
    ax : matplotlib.Axes
        The Axes object on which the data should be plotted.
    axtype : string ("inner" or "outer")
        The position of the Axes in its row/col.
    col : int
        Which column the Axes occupies.
    xticks : bool
        Whether the xtick should be kept (True).
    ylabel : string or None
        Label for the y-axis.
    sublabel : string or None
        The sub-figure label for the Axes (e.g., "A" for the upper left
        Axes).
    labelsize : int
        The fontsize in points of the labels.

    """

    # common stuff
    if sublabel is None:
        sublabel = ''

    seaborn.despine(
        ax=ax,
        left=(col == 1 and axtype == 'outer') or axtype == 'inner',
        right=(col == 0 and axtype == 'outer') or axtype == 'inner',
        top=True,
        bottom=not ax.is_last_row()
    )

    # left side of the figure
    if axtype == 'outer' and col == 0:
        # remove ticks from the right
        ax.yaxis.tick_left()
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis='y', right='off', which='both', labelsize=8)
        ax.annotate(sublabel, (0.0, 1.0), xytext=(5, -10),
                    xycoords='axes fraction', textcoords='offset points',
                    fontsize=8, zorder=20)
    # right side of the figure
    elif axtype == 'outer' and col == 1:
        ax.tick_params(axis='y', left='off', which='both', labelsize=8)

        # remove ticks from the left
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(ylabel, rotation=270, fontsize=8)
        #label = ax.yaxis.get_label()
        #label.set_rotation(270)
        ax.annotate(sublabel, (1.0, 1.0), xytext=(-20, -10),
                    xycoords='axes fraction', textcoords='offset points',
                    fontsize=8, zorder=20)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', right='off', left='off', which='both')


    # clear tick labels if xticks is False or Nones
    if not xticks:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    else:
        ax.tick_params(axis='x', labelsize=7)
