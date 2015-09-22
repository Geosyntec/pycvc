import os

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


__all__ = [
    'WQItem',
    'WQComparison',
    'WQMegaFigure',
]


class WQItem(object):
    """ A minimal representation of the water quality results and storms
    at a particular site, for a particular pollutant.

    Parameters
    ----------
    siteobj : cvc.dataAccess.Site Object
        Object representating all of the data for a CVC monitoring site.
    sampletype : string
        The type of samples to be analyzed. Valid values are "grab" and
        "composite".
    parametername : string
        The official name of the pollutant to be analyzed. See the
        database or ``cvc.info``.

    """

    def __init__(self, siteobj, sampletype, parametername):

        # basic atrributes
        self.site = siteobj
        self.siteid = self.site.siteid

        # sample type (grab or comp)
        self.sampletype = dataAccess._check_sampletype(sampletype)
        self.parametername = parametername

        # properties to be lazily-loaded when needed.
        self._data = None
        self._number_detects = None
        self._number_nondetects = None
        self._parameter = None
        self._location = None
        self._load_location = None

    @property
    def data(self):
        """ The main dataframe with the WQ and storm data. """
        if self._data is None:
            stype = self.sampletype
            param = self.parametername
            self._data = self.site.tidy_data.query(
                "parameter == @param and sampletype == @stype"
            )
        return self._data

    @property
    def mindate(self):
        """ The first sample collected """
        return self.data['samplestart'].min()

    @property
    def maxdate(self):
        """ The last sample collected """
        return self.data['samplestop'].max()

    @property
    def parameter(self):
        """ wqio.Parameter object for the data """
        if self._parameter is None:
            POC = list(filter(
                lambda p: p['cvcname'] == self.parametername, info.POC_dicts
            ))[0]

            # xxx: units (hack hack hack)
            _units = self.data['units'].unique()
            if _units.shape[0] > 1 or _units[0] != POC['conc_units']['plain']:
                raise ValueError("inconsistent units for {} ({})".format(self.parametername, _units))

            self._parameter = wqio.Parameter(
                name=POC['cvcname'],
                units=POC['conc_units']['tex'],
                usingTex=True
            )
        return self._parameter

    @property
    def number_detects(self):
        """ Number of concentration values above the detection limit """
        if self._number_detects is None:
            self._number_detects = self.data.query("qualifier == '='").shape[0]
        return self._number_detects

    @property
    def number_nondetects(self):
        """ Number of concentration values below the detection limit """
        if self._number_nondetects is None:
            self._number_nondetects = self.data.shape[0] - self.number_detects
        return self._number_nondetects

    @property
    def location(self):
        """ wqio.Location representation of the concentration data """
        if self._location is None:
            self._location = wqio.Location(self.data, rescol='concentration',
                                           qualcol='qualifier', ndval='<',
                                           station_type='outflow')
            self._location.name = self.siteid
            self._location.color = self.site.color
            self._location.plot_marker = self.site.marker
        return self._location

    @property
    def load_location(self):
        """ wqio.Location representation of the pollutant load data """
        if self._load_location is None:
            rescol = 'load_outflow'
            df = self.data.dropna(subset=[rescol]).query("load_outflow > 0")
            self._load_location = wqio.Location(df,
                                           rescol=rescol,
                                           qualcol='qualifier',
                                           ndval='<',
                                           station_type='outflow',
                                           useROS=False)
            self._load_location.name = self.siteid
            self._load_location.color = self.site.color
            self._load_location.plot_marker = self.site.marker
        return self._load_location


class WQComparison(object):
    """ A container class for comparing mulitiple CVC site objects.

    Parameters
    ----------
    siteobj : List of CVCData objects
        List of objects representing the sites to compared to the
        control site and external data.
    refobj : CVCData object
        Object representing the control (untreated) site/
    sampletype : string
        The type of samples to be analyzed in the ``siteboj`` and
        ``refobj``. Valid values are 'grab' or 'composite'.
    parametername : string
        The pollutant of concern to summarize.
    nsqdata : cvc.external.nsqd object:
        Representation of Bob Pitt's National Stormwater Quality
        dataset.
    bmpdb : cvc.external.bmpdb object:
        Representation of the Internation Stormwater BMP Performance
        Database.

    """

    def __init__(self, siteobjects, sampletype, parametername, nsqdata, bmpdb):

        # list of CVCDate objects
        self.sites = siteobjects
        self.NSites = len(siteobjects)

        # sample type (grab or comp)
        self.sampletype = dataAccess._check_sampletype(sampletype)
        self.parametername = parametername
        self.cvcparameter = parametername

        # the CVC pollutant
        self.POC = list(filter(
                lambda p: p['cvcname'] == self.parametername, info.POC_dicts
        ))[0]

        # Parameter object for the BMP Database
        self.bmpparameter = wqio.Parameter(
            name=self.POC['bmpname'],
            units=self.POC['conc_units']['tex'],
            usingTex=True
        )

        # Parameter object for the NSQD
        self.nsqdparameter = wqio.Parameter(
            name=self.POC['nsqdname'],
            units=self.POC['conc_units']['tex'],
            usingTex=True
        )

        # external data objects
        self.nsqdata = nsqdata
        self.bmpdb = bmpdb

        # color scheme for seasonally summarized data
        self.seasonal_palette = "BrBG_r"

        # properties to be lazily-loaded later
        self._wqitems = None
        self._datalabels = None
        self._bmplabels = None
        self._landuselabels = None
        self._maxN = None
        self._max_detects = None
        self._siteid = None
        self._tidy = None
        self._parameter = None
        self._load_parameter = None

    @property
    def parameter(self):
        """ wqio.Parameter for the pollutant concentration data """
        if self._parameter is None:
            self._parameter = wqio.Parameter(
                name=self.POC['cvcname'],
                units=self.POC['conc_units']['tex'],
                usingTex=True
            )
        return self._parameter

    @property
    def load_parameter(self):
        """ wqio.Parameter for the pollutant load data """
        if self._load_parameter is None:
            _unit = self.POC['load_units']
            if _unit == 'g':
                units = r'\si[per-mode=symbol]{\gram}'
            else:
                units = _unit

            self._load_parameter = wqio.Parameter(
                name=self.POC['cvcname'] + ' Load',
                units=units,
                usingTex=True
            )
        return self._load_parameter

    @property
    def wqitems(self):
        """ List of the the WQItems to be compared """
        if self._wqitems is None:
            self._wqitems = [
                WQItem(site, self.sampletype, self.parametername)
                for site in self.sites
            ]

            # check that units are consisten.
            num_units = np.unique([wqcomp.parameter.units for wqcomp in self.wqitems]).shape[0]
            if num_units > 1:
                msg = 'Inconsistent units for {} {} samples'
                raise ValueError(msg.format(self.parametername, self.sampletype))

        return self._wqitems

    @property
    def datalabels(self):
        """ labels for all of the WQItems """
        if self._datalabels is None:
            # datalabels, markers, and colors
            self._datalabels = [item.siteid for item in self.wqitems]

        return self._datalabels

    @property
    def bmplabels(self):
        """ Labels for the BMP Database """
        if self._bmplabels is None:
            self._bmplabels = [item.siteid for item in self.wqitems]
            self._bmplabels.extend(bmpcats_to_use)
        return self._bmplabels

    @property
    def landuselabels(self):
        """ Labels for landuse data (NSQD) """
        if self._landuselabels is None:
            # loop through all of the landuses in NSQD
            self._landuselabels = [item.siteid for item in self.wqitems]
            self._landuselabels.extend(list(filter(
                lambda lu: lu.lower() != 'unknown',
                self.nsqdata.data['Primary Landuse'].unique()
            )))
        return self._landuselabels

    @property
    def maxN(self):
        """ Max number of results in all of the items """
        if self._maxN is None:
            self._maxN = max([w.data.shape[0] for w in self.wqitems])
        return self._maxN

    @property
    def max_detects(self):
        """ Max number of detects in all of the items """
        if self._max_detects is None:
            self._max_detects = max([w.number_detects for w in self.wqitems])
        return self._max_detects

    @property
    def siteid(self):
        """ Site ID of the primary site """
        if self._siteid is None:
            self._siteid = self.sites[0].siteid
        return self._siteid

    @property
    def tidy(self):
        """ Tidy representation of the WQ data """
        if self._tidy is None:
            self._tidy = pandas.concat([w.data for w in self.wqitems])
        return self._tidy

    def savefig(self, fig, figname, load=False):
        """ Conveniently save a matplotlib figures

        Parameters
        ----------
        fig : matplotlib.Figure
            The figrue to be saved
        figname : string
            The filename without the extension.
        load : bool, optional (default = False)
            If True, appends "-load" to the end of ``figname``.
        """

        viz._savefig(fig, figname, extra='Individual', load=load)

    def locations(self, load=False):
        """ Returns a list of wqio.Locations for the data.

        Parameters
        ----------
        load : bool, optional (default = False)
            Toggles if pollutant concentration or load data should be
            use to contruct the Location objects.

        Returns
        -------
        locations : list of wqio.Location objects

        """

        if load:
            return [wqi.load_location for wqi in self.wqitems]
        else:
            return [wqi.location for wqi in self.wqitems]

    def ylabel(self, load=False):
        """ Returns a label for the y-axis of a figure.

        Parameters
        ----------
        load : bool, optional (default = False)
            Toggles if considering pollutant concentration or load data.

        Returns
        -------
        ylabel : string

        """
        if load:
            return self.load_parameter.paramunit(usecomma=True)
        else:
            return self.parameter.paramunit(usecomma=True)

    def describe(self):
        """ Returns a basic statistics of the data.

        Parameters
        ----------
        None

        Returns
        -------
        stat : pandas.DataFrame

        """
        stats = (
            self.tidy.groupby(by=['site'])['concentration']
                     .apply(lambda g: g.describe())
                     .unstack(level='site')
        )

        for wqi in self.wqitems:
            stats.loc['ND', wqi.siteid] = wqi.number_nondetects

        return stats

    def compareResults(self):
        raise NotImplementedError

        siteids = [site.siteid for site in self.sites]
        filenames = [
            utils.processFilename(
                '%s_%s_%s.csv' %
                (wq.siteid, self.sampletype.title(), self.bmpparameter.name.title())
            ) for wq in self.wqitems
        ]

        filepaths = [os.path.join('..', 'output', 'csv', name) for name in filenames]

        # construct a new filename
        outputfilename = utils.processFilename('%sWQComparison_%s_%s' %
                                               (self.siteid, self.sampletype.title(),
                                               self.bmpparameter.name.title()))
        # full paths for the output
        csvpath = os.path.join('cvc', 'output', 'csv', outputfilename + '.csv')
        texpath = os.path.join('cvc', 'output', 'tex', outputfilename + '.tex')

        data = _compare_files(filepaths, siteids, csvpath, texpath)

        return data

    def compareStats(self):
        raise NotImplementedError
        '''
        Makes a LaTeX table and CSV file of the statistical summary if a site and
            Control site.

        Input:
            None

        Writes:
            CSV and LaTeX tables of the stat summaries

        Returns:
            ssumary (pandas dataframe) : a joined dataframe of both summaries
        '''
        # map of row numbers to row headers, stat summary attributes,
        # and numerical formats
        stat_dict = {
            0: ('Count', 'N', '%d'),
            1: ('Count of non-detects', 'ND', '%d'),
            2: ('Mean', 'mean', '%0.2f'),
            3: ('Std. dev.', 'std', '%0.2f'),
            4: ('Min.', 'min', '%0.2f'),
            5: ('25th percentile', 'pctl25', '%0.2f'),
            6: ('Median', 'median', '%0.2f'),
            7: ('75th percentile', 'pctl75', '%0.2f'),
            8: ('Max.', 'max', '%0.2f')
        }

        columns = ['Statistic']
        for wqitem in self.wqitems:
            columns.append(wqitem.siteid)

        # make an empty data frame for the summaries
        stat_table = pandas.DataFrame(data=np.empty((9, len(columns)),
                                      dtype='S50'), columns=columns)

        # loop through the stat map and populate the dataframe
        for stat in stat_dict:
            stat_table.loc[stat, 'Statistic'] = stat_dict[stat][0]

            for wqitem in self.wqitems:
                if wqitem.stats is not None:
                    stat_table.loc[stat, wqitem.siteid] = stat_dict[stat][2] % \
                        getattr(wqitem.stats, stat_dict[stat][1])
                else:
                    stat_table.loc[stat, wqitem.siteid] = '--'

        # make the file names and paths
        filename = utils.processFilename('%s_StatSummary_%s_%s' %
                                         (self.siteid, self.sampletype.title(),
                                          self.bmpparameter.name.title()))
        csvpath = os.path.join('cvc', 'output', 'csv', filename + '.csv')
        stat_table.to_csv(csvpath, index=False, na_rep='--')

        # write the tex file
        texpath = os.path.join('cvc', 'output', 'tex', filename + '.tex')
        utils.csvToTex(csvpath, texpath)

        # write the csv
        return stat_table

    def _add_std_hline(self, ax):
        """ Adds a horizontal line to an axis at the WQ guideline for a
        given parameter

        Parameters
        ----------
        ax : matplotlib.Axes
            The Axes on which the horizontal line should be drawn.

        Returns
        -------
        None
        """

        if self.parameter.name:
            param = self.parameter.name
            # get the standard for the parameter from each site object
            standards = []
            for site in self.sites:
                subset = site.wqstd.query("parameter == @param")['upper_limit']
                if subset.shape[0] > 0:
                    standards.append(subset.iloc[0])

            # ensure that all of the standards are the same
            standards = np.unique(standards)
            if standards.shape[0] > 1:
                raise ValueError('Guidelines for %s are inconsistent' % self.parameter.name)
            elif standards.shape[0] == 1:
                # convert to a float value
                try:
                    std = float(standards[0])
                except:
                    raise ValueError('Could not parse guideline for %s' % self.parameter.name)
            else:
                std = np.nan

            # plot it
            if std is not None:
                ax.axhline(y=std, linewidth=2, color='Coral', zorder=0,
                           label='PWQO/CCME Guideline')

    def _do_boxplots(self, ax1, position, showstandard=True, load=False):
        """ boxplot helper function """
        # make boxplots of the site and Control data
        xlabels = []
        for pos, loc in enumerate(self.locations(load=load), position):
            loc.boxplot(ax=ax1, pos=pos, showmean=True, patch_artist=True,
                        notch=not load)
            xlabels.append(loc.name)

        # add the  guideline
        if showstandard and not load:
            self._add_std_hline(ax1)
        return pos, xlabels

    def _do_probplots(self, ax2, showstandard=True, load=False):
        """ probability plot helper function """
        n = []
        for loc in self.locations(load=load):
            loc.probplot(ax=ax2)
            n.append(loc.N)
        N = max(n)
        ax2.set_xlim(left=0.5, right=99.5)

        # add the guideline
        if showstandard and not load:
            self._add_std_hline(ax2)

    def _do_timeseries(self, ax, showstandard=True, load=False):
        """ time series helper function """
        if load:
            rescol = 'load_outflow'
        else:
            rescol = 'concentration'
        for wqi in self.wqitems:
            # try to plot stuff and keep track of counts
            N, ND = 0, 0

            detects = wqi.data['qualifier'] != '<'
            nondetects = wqi.data['qualifier'] == '<'

            # site data detects
            N = viz._try_to_plot_ts(
                wqi.data.query("qualifier != '<'"),
                rescol=rescol, datecol='samplestart',
                ax=ax, marker=wqi.site.marker, markersize=4,
                markerfacecolor=wqi.site.color, linestyle='none',
                markeredgecolor='k', zorder=4, alpha=0.5,
                label=wqi.site.siteid
            )

            # site data detects
            ND = viz._try_to_plot_ts(
                wqi.data.query("qualifier == '<'"),
                rescol, 'samplestart',
                ax=ax, marker=wqi.site.marker, markersize=4,
                markerfacecolor='none', linestyle='none',
                markeredgecolor=wqi.site.color, zorder=4, alpha=0.5,
                label=r'nolegend'
            )

        # add the guideline
        if showstandard and not load:
            self._add_std_hline(ax)
        return N, ND

    def parameterTimeSeries(self, ax=None, addLegend=False, addNote=True,
                            finalOutput=True, showstandard=True, load=False):
        """ Plot time series of a parameter at a site and Control site.

        Parameters
        ----------
        ax : matplotlib.Axes or None
            Axes onto which the data should be plotted. If not provided,
            a new plot will be created.
        addLegend : bool, optional (default = False)
            Toggles adding a legend to ``ax``.
        addNote : bool, optional (default = True)
            Toggles including the note explaining the symbology of
            non-detect results.
        finalOutput : bool, optional (default = True)
            When True, the ``ax`` is cleane up and formatted to be
            saved.
        showstandard : bool, optional (default = True)
            When True, the WQ guideline is drawn as a horizontal line
            across the ``ax``.
        load : bool, optional (default = False)
            When True, indicates that the data represent pollutant loads
            instead if concentrations.

        Returns
        -------
        fig : matplotlib.Figure

        """

        # check the Axes object
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(figsize=(6, 2.75))

        # plot the time series
        N, n = self._do_timeseries(ax, showstandard=showstandard, load=load)

        # if detect site data actually got plotted
        if N > 0:
            # axes labels and scales
            ax.set_xlabel('')
            ax.set_ylabel(self.ylabel(load=load), size=8)
            ax.set_yscale('log')

            if addLegend:
                # legend stuff
                leg = ax.legend(fontsize=7, bbox_to_anchor=(0.75, 1.115),
                                handletextpad=0.4, columnspacing=0.5,
                                ncol=self.NSites)
                leg.get_frame().set_alpha(0.80)
                leg.get_frame().set_zorder(25)

            if addNote:
                # add the note
                note = 'Note: Open symbols indicate non-detect results'
                ax.annotate(note, (0.01, 1.03), xycoords='axes fraction',
                            fontsize=7)

            # axes limits, padded by 5 days
            axpad = pandas.offsets.Day(5)
            xmax = np.max([wqi.maxdate for wqi in self.wqitems]) + axpad
            xmin = np.min([wqi.mindate for wqi in self.wqitems]) - axpad
            ax.set_xlim([xmin.toordinal(), xmax.toordinal()])

            # tick label formats
            days = mdates.DayLocator(bymonthday=range(0, 32, 7))
            ax.xaxis.set_minor_locator(days)
            monthfmt = mdates.DateFormatter('%b\n%Y')
            ax.xaxis.set_major_formatter(monthfmt)
            ax.tick_params(reset=True)
            ax.tick_params(axis='both', labelsize=8)

            # ylabel tick labels
            label_format = mticker.FuncFormatter(utils.figutils.alt_logLabelFormatter)
            ax.yaxis.set_major_formatter(label_format)
            utils.figutils.gridlines(ax, yminor=True, xminor=False)

            if finalOutput:
                # tick selection
                ax.xaxis.tick_bottom()
                ax.yaxis.tick_left()

                # optimize the layour
                fig.tight_layout()
                seaborn.despine(ax=ax)

                # make the filename and path
                figname = utils.processFilename(
                    '%s_TS_%s_%s' % (self.siteid, self.sampletype.title(),
                                     self.parameter.name.title())
                )
                self.savefig(fig, figname, load=load)

        return fig

    def parameterStatPlot(self, ax1=None, ax2=None, labelax1=True,
                          finalOutput=False, showstandard=False,
                          load=False):
        """ Make figure of side-by-side boxplots and probability plots
        comparing site and Control data.

        Parameters
        ----------
        ax1, ax2 : matplotlib.Axes or None
            Axes onto which the data should be plotted. If not provided,
            a new plot will be created. Values must be of the same type.
            Boxplots are drawn onto ``ax1``. Probability plots are drawn
            on ``ax2``
        labelax1 : bool, optional (default = True)
            When True, a y-axis label is added to ``ax1`` (boxplots).
        finalOutput : bool, optional (default = True)
            When True, the ``ax`` is cleane up and formatted to be
            saved.
        showstandard : bool, optional (default = True)
            When True, the WQ guideline is drawn as a horizontal line
            across the ``ax``.
        load : bool, optional (default = False)
            When True, indicates that the data represent pollutant loads
            instead if concentrations.

        Returns
        -------
        fig : matplotlib.Figure

        """

        if self.max_detects >= 5:

            if (ax1 is None) != (ax2 is None):
                raise ValueError("ax1, ax2 must both be axes objects or None")

            elif ax1 is None and ax2 is None:
                # create the figure (width of boxplot = 1, width of probplot = 3)
                # TODO: turn this into a utils.figutils function
                fig = plt.figure(figsize=(6.0, 2.5))
                ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=1)
                ax2 = plt.subplot2grid((1, 4), (0, 1), colspan=3)

            else:
                fig = ax1.figure

            # boxplots
            position, xlabels = self._do_boxplots(ax1, 1, load=load, showstandard=showstandard)
            ax1.set_xlim(left=0.5, right=position+0.5)
            ax1.set_xticks(np.arange(1, position+1))
            label_options = dict(rotation=35, rotation_mode='anchor', ha='right', fontsize=8)
            ax1.set_xticklabels(xlabels, **label_options)

            # prob plots
            self._do_probplots(ax2, load=load, showstandard=showstandard)

            # boxplot tick labels
            ax1.tick_params(axis='x', labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)


            if labelax1:
                ax1.set_ylabel(self.ylabel(load=load), size=8)

            if finalOutput:
                # tick selection
                ax1.xaxis.tick_bottom()
                ax1.yaxis.tick_left()
                ax2.xaxis.tick_bottom()
                ax2.yaxis.tick_right()
                ax2.set_xlabel(r'Non-exceedance Probability (\%)')

                # layout everything all nice
                seaborn.despine(ax=ax1)
                seaborn.despine(ax=ax2, left=True, right=False)
                fig.tight_layout()

                # manual layout touch: spacing between axes
                fig.subplots_adjust(wspace=0.03)

                # fle names and saving figures
                figname = utils.processFilename(
                    '{}_Statplot_{}_{}'.format(
                        self.siteid, self.sampletype.title(),
                        self.parameter.name.title()
                    )
                )
                self.savefig(fig, figname, load=load)
            return fig

    def _external_boxplots(self, which, ax=None, addNote=True,
                           finalOutput=False, showstandard=True):
        """ Create boxplots of site and control data alongside data from
        the BMPDB grouped by BMP Category or NSQD grouped by landuse.

        Parameters
        ----------
        which : str
            The name of the external dataset. Valid values are
            'BMPDB' or 'NSQD'.
        ax : matplotlib.Axes or None
            Axes onto which the data should be plotted. If not provided,
            a new plot will be created.
        addNote : bool, optional (default = True)
            Toggles including the note explaining the origin of the
            external data.
        finalOutput : bool, optional (default = True)
            When True, the ``ax`` is cleane up and formatted to be
            saved.
        showstandard : bool, optional (default = True)
            When True, the WQ guideline is drawn as a horizontal line
            across the ``ax``.

        Returns
        -------
        xlabels : list of x-tick labels.

        """

        # get properties of the BMP DB dataset (name of the table it's in,
        # how the name is spelled etc)
        if which.lower() == 'bmpdb':
            external = self.bmpdb
            key = 'bmpname'
            note = 'Note: BMP category data from BMP BD'
            plot_type = 'BMPType'
        elif which.lower() == 'nsqd':
            external = self.nsqdata
            key = 'nsqdname'
            note = 'Note: Land use data from NSQD (Pitt, 2008)'
            plot_type = 'Landuse'
        else:
            raise ValueError("which must be either, 'BMPDB' or 'NSQD'")

        extparamname = info.getPOCInfo('cvcname', self.cvcparameter, key)
        selection_kwds = dict(squeeze=True, parameter=extparamname, station='outflow')

        # if the parameter exists and there's enough site data (N >= 5)
        if self.max_detects >= 5:

            # setup the figure, x-tick labels, boxplot positions
            if ax is not None:
                fig = ax.figure
            else:
                fig, ax = plt.subplots(figsize=(6.5, 4.0))

            position = 1

            # site and Control boxplots
            position, xlabels = self._do_boxplots(ax, position, showstandard=showstandard)

            if addNote:
                # add the note
                ax.annotate(note, (0.01, 1.03), xycoords='axes fraction', fontsize=7)

            if extparamname is not None:
                position, xlabels = external.boxplot(ax, position, xlabels, **selection_kwds)

            ax.set_ylabel(self.ylabel(), size=8)
            ax.set_xlim(left=0.5, right=position+0.5)
            ax.set_xticks(np.arange(1, position+1))
            label_options = dict(rotation=35, rotation_mode='anchor', ha='right', fontsize=8)
            ax.set_xticklabels(xlabels, **label_options)

            if finalOutput:

                # tick selection
                ax.xaxis.tick_bottom()
                ax.yaxis.tick_left()

                # overall layout
                seaborn.despine(ax=ax)
                fig.tight_layout()

                # figure name and saving
                figname = utils.processFilename(
                    '{}_Boxplot_{}_{}_{}'.format(
                        self.siteid, plot_type, self.sampletype.title(),
                        self.parameter.name.title()
                    )
                )
                self.savefig(fig, figname, load=False)

        return xlabels

    def bmpCategoryBoxplots(self, ax=None, addNote=True,
                            finalOutput=False, showstandard=True):
        """ Create boxplots of site and control data alongside data from
        the BMPDB grouped by BMP Category.

        Parameters
        ----------
        ax : matplotlib.Axes or None
            Axes onto which the data should be plotted. If not provided,
            a new plot will be created.
        addNote : bool, optional (default = True)
            Toggles including the note explaining the origin of the
            external data.
        finalOutput : bool, optional (default = True)
            When True, the ``ax`` is cleane up and formatted to be
            saved.
        showstandard : bool, optional (default = True)
            When True, the WQ guideline is drawn as a horizontal line
            across the ``ax``.

        Returns
        -------
        xlabels : list of x-tick labels.

        """
        return self._external_boxplots('bmpdb', ax=ax, addNote=addNote,
                                       finalOutput=finalOutput,
                                       showstandard=showstandard)

    def landuseBoxplots(self, ax=None, addNote=True,
                        finalOutput=False, showstandard=True):
        """ Create boxplots of site and control data alongside data from
        the NSQD grouped by landuse.

        Parameters
        ----------
        ax : matplotlib.Axes or None
            Axes onto which the data should be plotted. If not provided,
            a new plot will be created.
        addNote : bool, optional (default = True)
            Toggles including the note explaining the origin of the
            external data.
        finalOutput : bool, optional (default = True)
            When True, the ``ax`` is cleane up and formatted to be
            saved.
        showstandard : bool, optional (default = True)
            When True, the WQ guideline is drawn as a horizontal line
            across the ``ax``.

        Returns
        -------
        xlabels : list of x-tick labels.

        """

        return self._external_boxplots('nsqd', ax=ax, addNote=addNote,
                                       finalOutput=finalOutput,
                                       showstandard=showstandard)

    def seasonalBoxplots(self, ax=None, finalOutput=False,
                         showstandard=True, load=False):
        """ Create boxplots of site and control data grouped by season.

        Parameters
        ----------
        ax : matplotlib.Axes or None
            Axes onto which the data should be plotted. If not provided,
            a new plot will be created.
        finalOutput : bool, optional (default = True)
            When True, the ``ax`` is cleane up and formatted to be
            saved.
        showstandard : bool, optional (default = True)
            When True, the WQ guideline is drawn as a horizontal line
            across the ``ax``.
        load : bool, optional (default = False)
            When True, indicates that the data represent pollutant loads
            instead if concentrations.

        Returns
        -------
        fig : matplotlib.Figure

        """
        seasons = ['winter', 'spring', 'summer', 'autumn']
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(figsize=(6.5, 4.0))

        if load:
            rescol = 'load_outflow'
        else:
            rescol = 'concentration'

        ax = seaborn.boxplot(x='site', y=rescol, data=self.tidy, ax=ax,
                             hue='season', hue_order=seasons,
                             palette=self.seasonal_palette,
                             showcaps=False, medianprops=dict(lw=1.),
                             width=0.75)

        ax.set_yscale('log')
        ax.set_ylabel(self.ylabel(load=load))
        ax.yaxis.grid(False)
        ax.set_xlabel('')

        if showstandard and not load:
            self._add_std_hline(ax)

        if finalOutput:
            # tick selection
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()

            # optimize the layout
            seaborn.despine(ax=ax)
            fig.tight_layout()

            # make the filename and path
            figname = utils.processFilename(
                '%s_Seasonal_%s_%s' % (self.siteid, self.sampletype.title(),
                                 self.parameter.name.title())
            )
            self.savefig(fig, figname, load=load)

        return fig


class WQMegaFigure(object):
    """ Simple contraine to effectively create summary plots from
    WQComparison objects.

    Parameters
    ----------
    siteobjects : list of cvc.Site objects
        Sites that should be compared.
    sampletype : string
        Type of samples whose reports should be generated. Valid values
        are "grab" or "composite". Only "composite" is recommended.
    parameterlist : list of strings
        List of the CVC names for the parameters to be summarized.
    fignum : int
        The number of the figure e.g., 1, 2...
    nsqddata : cvc.external.nsqd object
        Data structure representing the National Stormwater Quality
        Dataset.
    bmpdb : cvc.external.nsqd object
        Data structure representing the Internation Stormwater BMP
        Database.

    """

    def __init__(self, siteobjects, sampletype, parameterlist, fignum, nsqdata, bmpdb):
        self.sites = siteobjects
        self.sampletype = sampletype
        self.parameterlist = parameterlist
        self.nrows = np.ceil(len(self.parameterlist)/2.)
        self.figsize = (6.5, 7)
        self.fignum = fignum
        self._nsqdata = nsqdata
        self._bmpdb = bmpdb

        self.wqcomparisons = [
            WQComparison(siteobjects, sampletype, param, nsqdata, bmpdb)
            for param in parameterlist
        ]

        self.siteid = self.wqcomparisons[0].siteid

    def _basic_plotter(self, plotfxn, figname, figshape=(4, 2), startcols=(0,),
                       externaldata=None, globalxlimits=True, removelegend=True,
                       seasonal=False, load=False):

        fig = plt.figure(figsize=self.figsize, facecolor='none', edgecolor='none')
        fig_gs = gridspec.GridSpec(*figshape)

        letters = list('abcdefghijklmnop')
        all_axes = []
        for n, wqcomp in enumerate(self.wqcomparisons):
            ax_gs = gridspec.GridSpecFromSubplotSpec(
                1, 4, subplot_spec=fig_gs[n], wspace=0.00
            )

            sublabel = '(%s)' % letters[n]
            col = n % 2
            addNote = (n == 0)

            axes = []
            for colnum in range(len(startcols)):
                if colnum < len(startcols) - 1:
                    cspan_start = startcols[colnum]
                    cspan_stop = startcols[colnum+1]
                    axes.append(fig.add_subplot(ax_gs[cspan_start:cspan_stop]))
                else:
                    axes.append(fig.add_subplot(ax_gs[startcols[colnum]:]))

            plotfxn(wqcomp, axes, addNote=addNote, load=load)
            all_axes.append(axes)

            #if row < self.nrows - 1:
            if n >= len(self.wqcomparisons) - 2:
                xticks = True
            else:
                xticks = False

            ylabel = wqcomp.ylabel(load=load)

            if len(axes) == 1:
                viz.formatGSAxes(axes[0], 'outer', col, xticks, ylabel,
                             sublabel=sublabel, labelsize=7)

            elif len(axes) == 2:
                viz.formatGSAxes(axes[col], 'outer', col, xticks, ylabel,
                             sublabel=sublabel, labelsize=7)

                viz.formatGSAxes(axes[col-1], 'inner', col, xticks, ylabel,
                             labelsize=7)

            else:
                raise NotImplementedError('no more than 2 axes per gridspec')


        # unify all of the axes x-limits
        if globalxlimits:
            globalmin = np.min([ax.get_xlim()[0] for ax in fig.get_axes()])
            globalmax = np.max([ax.get_xlim()[1] for ax in fig.get_axes()])
            for ax in fig.get_axes():
                ax.set_xlim(left=globalmin, right=globalmax)

        for n, ax in enumerate(np.array(all_axes).flat):
            leg = ax.get_legend()
            if (n > 0 or removelegend) and leg is not None:
                ax.legend_.remove()

        # turn on the legend using the last axes
        leg = self._proxy_legend(all_axes[1][-1], externaldata=externaldata,
                                 seasonal=seasonal, load=load)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.12, top=0.96, wspace=0.04)

        figname = utils.processFilename('%s-megafig-%s-%s-%02d' %
                                        (self.siteid, figname, self.sampletype,
                                         self.fignum))
        self.savefig(fig, figname, load=load)
        return fig, all_axes

    def _proxy_legend(self, ax, externaldata=None, seasonal=False, load=False):
        artists = []
        labels = []
        wqcomp = self.wqcomparisons[0]

        if seasonal:
            labels = ['winter', 'spring', 'summer', 'autumn']
            palette = seaborn.color_palette(wqcomp.seasonal_palette, n_colors=4)
            for color, season in zip(palette, labels):
                line = mlines.Line2D([0], [0], marker='s', markersize=8, color=color,
                                     linestyle='none', markeredgecolor='k')
                artists.append(line)

        else:
            for site in self.sites:
                line = mlines.Line2D([0], [0], marker=site.marker, markersize=4,
                                     linestyle='none', color=site.color)
                artists.append(line)
                labels.append(site.siteid)

            if externaldata is not None:
                if externaldata.lower() == 'nsqd':
                    external = self._nsqdata
                elif externaldata.lower() == 'bmpdb':
                    external = self._bmpdb

                artists.append(mlines.Line2D([0], [0],
                               marker=external.marker,
                               markersize=4, linestyle='none',
                               color=external.color))
                labels.append(externaldata)
        if not load:
            artists.append(mlines.Line2D([0], [0], linewidth=2, color='Coral'))
            labels.append('PWQO/CCME Guideline')

        ncol = int(np.ceil(len(artists)/2.0))
        leg = ax.legend(artists, labels, fontsize=6, ncol=ncol,
                        frameon=False, bbox_to_anchor=(1.0, 1.23))
        return leg

    def savefig(self, fig, figname, load=False):
       viz._savefig(fig, figname, extra='Megafigure', load=load)

    def timeseriesFigure(self, load=False):
        """ Creates time series plots for each pollutant

        Parameters
        ----------
        load : bool, optional (default = False)
            Indicates if the values plotted are loads.

        Returns
        -------
        None

        """

        fig, axes = self._basic_plotter(viz._make_time_series, 'timeseries', startcols=(0,),
                                        load=load)

    def bmpCategoryBoxplotFigure(self):
        """ Creates boxplots comparing CVC sites to BMP Database
        categories.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        fig, axes = self._basic_plotter(viz._make_bmp_boxplot, 'bmpBoxplot',
                                        startcols=(0,), externaldata='BMPDB')

    def landuseBoxplotFigure(self):
        """ Creates boxplots comparing CVC sites to NSQD landuse types.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        fig, axes = self._basic_plotter(viz._make_landuse_boxplot, 'landuseBoxplots',
                                        startcols=(0,), externaldata='NSQD')

    def statplotFigure(self, load=False):
        """ Creates boxplots and probability plots comparing CVC sites.

        Parameters
        ----------
        load : bool, optional (default = False)
            Indicates if the values plotted are loads.

        Returns
        -------
        None

        """

        fig, axes = self._basic_plotter(viz._make_statplots, 'statplot',startcols=(0, 1),
                                        globalxlimits=False, load=load)

    def seasonalBoxplotFigure(self, load=False):
        """ Creates boxplots comparing CVC sites on a seasonal basis.

        Parameters
        ----------
        load : bool, optional (default = False)
            Indicates if the values plotted are loads.

        Returns
        -------
        None

        """
        fig, axes = self._basic_plotter(viz._make_seasonal_boxplot, 'seasonalBoxplots',
                                        startcols=(0,), removelegend=True, seasonal=True,
                                        load=load)


@np.deprecate
class SummaryAppendix(object):
    def __init__(self, siteobjects, sampletype, parameterdict, paramgrouplist,
                 inputfilename, outputfilename, version='draft'):
        self.sites = siteobjects
        self.sampletype = sampletype
        self.parameterdict = parameterdict
        self.paramgrouplist = paramgrouplist
        self.version = version
        self.inputfilename = inputfilename
        self.outputfilename = outputfilename
        self.templatefile = 'cvc/%s_appendix_template.tex' % version

    def makeTexInputFile(self, refsource):
        outputpath = os.path.join('cvc', 'output', 'tex', self.outputfilename)
        outputfile = open(outputpath, 'w')
        template = open(self.templatefile, 'r')
        tempstr = template.read()
        tempstr = tempstr.replace('__TEXFILE__', self.inputfilename)
        outputfile.write(tempstr)

        outputfile.close()
        template.close()

        inputpath = os.path.join('cvc', 'output', 'tex', self.inputfilename)
        with open(inputpath, 'w') as texfile:
            for site in self.sites:
                texfile.write('\n\\section{%s Water Quantity Data}\n' % site.tocentry)
                summarizeWaterQuantity(texfile, site, 'composite')

                texfile.write(
                    '\n\\section{%s Water Quality Data}\nNote: '
                    'Estimated volumes are determined by a Simple '
                    'Method transformation of precipitation data. '
                    'Estimated loads assume the median concentration '
                    'from the %s.' % (site.tocentry, refsource)
                )
                for group in dataAccess.groups:
                    summarizeWaterQuality(texfile, site, 'composite', group)

            texfile.write('\n\\section{Water Quality Performance Evaluation}')

            # include landuse figures
            landuse_section = 'Comparative Box Plots of Site and ' \
                              'Reference Data Categorized by Land Use'
            landuse_figtype = 'landuseBoxplots'
            landuse_caption = '\\textbf{Comparative Box Plots of Effluent Concentrations from ' \
                              'Site and Reference Data Categorized by Land Use }'
            summarizeWQComps(texfile, self.sites[0], self.sampletype,
                             landuse_section, landuse_figtype, landuse_caption)

            # include bmptype figures
            bmptype_section = 'Comparative Box Plots of Site and ' \
                              'Reference Data Categorized by BMP Type'
            bmptype_figtype = 'bmpBoxplot'
            bmptype_caption = '\\textbf{Comparative Box Plots of Effluent Concentrations from ' \
                              'Site and Reference Data Categorized by BMP Type }'
            summarizeWQComps(texfile, self.sites[0], self.sampletype,
                             bmptype_section, bmptype_figtype, bmptype_caption)

            # include time series figures
            ts_section = 'Time Series Plots of Water Quality ' \
                         'Sampling Events of Site and Control Data'
            ts_figtype = 'timeseries'
            ts_caption = '\\textbf{Time Series Plots of Site and Control Effluent ' \
                         'Concentrations from Sampling Events }'
            summarizeWQComps(texfile, self.sites[0], self.sampletype,
                             ts_section, ts_figtype, ts_caption)

            # include statplot figures
            sp_section = 'Graphical Statistical Summary Plots of Site and ' \
                         'Control Data'
            sp_figtype = 'statplot'
            sp_caption = '\\textbf{Box and Probability Plots of Site and Control Effluent Concentrations ' \
                         'from Sampling Events }'
            summarizeWQComps(texfile, self.sites[0], self.sampletype,
                             sp_section, sp_figtype, sp_caption)


@np.deprecate
class WQSummaryByParameter(object):

    def __init__(self, cvcdata, sampledate, parameter, resmedian):
        '''
        Object that represents the summary of the water quality results for a
            single sample of a parameter at site.

        Input:
            cvcdata (CVCData object) : the object representing in the entire
                dataset
            sampledate (pandas timestamp) : date and time the sample was
                collected
            parameter (string) : the parameter we're summarizing

        Writes:
            None

        Atrributes
            storm (Storm object) : object representing the summarized storm
            parameter (string) : parameter being summarized
            data (pandas DataFrame) : subset of water quality data for
                sample date and parameter
            conc (numpy array) : array of the concentration values
            load (numpy array) : array of the pollutant loads

        Methods (see docstrings):
            bigTableLine
        '''
        # make the storm and save teh parameter as an attribute
        self.siteid = cvcdata.siteid
        storm_number, self.storm = cvcdata._get_storm_from_date(sampledate)

        self.parameter = parameter

        # subset data
        self.sampledate = sampledate
        self.data = cvcdata.wqdata.xs([sampledate, parameter],
                                      level=['starttime', 'Parameter'])
        std = cvcdata.wqstd.xs(parameter)
        if std['guideline'] is not None and std['guideline'] != 'N/A':
            self.std_conc = std['guideline']
            if std['Units'] == 'ug/L':
                self.std_units = '(' + r'\si[per-mode=symbol]{\micro\gram\per\liter}' + ')'
            else:
                self.std_units = '(' + std['Units'] + ')'
        else:
            self.std_conc = '--'
            self.std_units = ''

        # if there's any data, return some concentrations and loads
        if self.data.shape[0] > 0:
            if self.data.Outflow_unit.values[0] == 'ug/L':
                self.conc_units = r'\si[per-mode=symbol]{\micro\gram\per\liter}'
            else:
                self.conc_units = self.data.Outflow_unit.values[0]

            self.conc = self.data.Outflow_res.values[0]
            unit_conversion = units_map[self.data.Outflow_unit.values[0]]

            if self.storm is not None:
                self.load_units = load_units[self.parameter]
                self.load = self.conc * self.storm.total_volume * unit_conversion
                self.infload = resmedian * self.storm.influent_volume * unit_conversion
            else:
                self.load = None
                self.load_units = None
                self.infload = None

        # `None` otherwise
        else:
            self.conc = None
            self.conc_units = None
            self.load = None
            self.load_units = None
            self.infload = None

    def bigTableLine(self):
        '''
        Creates a line to the big table summarizing WQ data for a storm.

        Input:
            None

        Writes:
            None

        Returns
            txt (string) : CSV string for a single row in the larger summary
                table
        '''
        effconc = utils.stringify(self.conc, '%s')
        precip = utils.stringify(self.storm, '%s', attribute='total_precip')
        infvol = utils.stringify(self.storm, '%s', attribute='influent_volume')
        effvol = utils.stringify(self.storm, '%s', attribute='total_volume')
        infload = utils.stringify(self.infload, '%s')
        effload = utils.stringify(self.load, '%s')

        if self.infload is not None and self.load is not None:
            loadreduction = self.infload - self.load
            loadreduction_percent = 100.0 * loadreduction / self.infload
        else:
            loadreduction = '--'
            loadreduction_percent = '--'

        if self.siteid != 'LV-1':
                  #'"Total Estimated\\footnotemark[1] Influent Volume (L)",' \
                  #'"Total Estimated\\footnotemark[1]\\footnotemark[2] Influent Load (%s)",' \
                  #'"Estimated\\footnotemark[1]\\footnotemark[2] Pollutant Load Reduction (%s)",' \
                  #'"Estimated\\footnotemark[1]\\footnotemark[2] Pollutant Load Reduction (%%)"\n' % \
            header = '"Date",' \
                '"Effluent EMC (%s)",' \
                '"Total Precipitation (mm)",' \
                '"Total Estimated Influent Volume (L)",' \
                '"Total Effluent Volume (L)",' \
                '"Total Estimated Influent Load (%s)",' \
                '"Total Effluent Load (%s)",' \
                '"Estimated Pollutant Load Reduction (%s)",' \
                '"Estimated Pollutant Load Reduction (%%)"\n' % \
                (self.conc_units, self.load_units, self.load_units, self.load_units)

            txt = '%s,%s,%s,%s,%s,%s,%s,%s,%s' % \
                (self.sampledate, effconc, precip, infvol, effvol, infload,
                    effload, loadreduction, loadreduction_percent)

        else:
                  #'"Total Estimated\\footnotemark[1] Effluent Volume (L)",' \
                  #'"Total Estimated\\footnotemark[1]\\footnotemark[2] Effluent Load (%s)"\n'  % \
            header = '"Date",' \
                '"Effluent EMC (%s)",' \
                '"Total Precipitation (mm)",' \
                '"Total Estimated Effluent Volume (L)",' \
                '"Total Estimated Effluent Load (%s)"\n' % \
                (self.conc_units, self.load_units)

            txt = '%s,%s,%s,%s,%s' % \
                (self.sampledate, effconc, precip, infvol, infload)

        return txt, header
