import os
import glob
import subprocess

import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as seaborn

from . import viz

import wqio
from wqio import utils


class LaTeXDirectory(utils.LaTeXDirectory):
    pass


class _WQSample_Mixin(wqio.core.samples._basic_wq_sample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._general_tex_table = None
        self._hydro_tex_table = None
        self._wq_tex_table = None
        self._storm_figure = None
        self._siteid = None
        self._tocentry = None
        self._wqstd = None
        self._templateISR = None
        self._documentISR = None

    @staticmethod
    def _res_with_units(res, units):
        if pandas.isnull(res):
            return '--'
        else:
            return '{} {}'.format(utils.sigFigs(res, 3), units)

    @property
    def siteid(self):
        return self._siteid
    @siteid.setter
    def siteid(self, value):
        self._siteid = value

    @property
    def general_tex_table(self):
        self._general_tex_table = utils.processFilename('{}-{}-1-General'.format(
            self.siteid, self.starttime.strftime('%Y-%m-%d-%H%M')
        ))
        return self._general_tex_table

    @property
    def hydro_tex_table(self):
        self._hydro_tex_table = utils.processFilename('{}-{}-2-Hydro'.format(
            self.siteid, self.starttime.strftime('%Y-%m-%d-%H%M')
        ))
        return self._hydro_tex_table

    @property
    def wq_tex_table(self):
        self._wq_tex_table = utils.processFilename('{}-{}-3-WQ_{}'.format(
            self.siteid, self.starttime.strftime('%Y-%m-%d-%H%M'),
            self.label.split(' ')[0])
        )
        return self._wq_tex_table

    @property
    def storm_figure(self):
        self._storm_figure = utils.processFilename('{}-{}-{}'.format(
            self.siteid, self.starttime.strftime('%Y-%m-%d-%H%M'),
            self.label.split(' ')[0])
        )
        return self._storm_figure

    @property
    def templateISR(self):
        return self._templateISR
    @templateISR.setter
    def templateISR(self, value):
        self._templateISR = value

    @property
    def tocentry(self):
        return self._tocentry
    @tocentry.setter
    def tocentry(self, value):
        self._tocentry = value

    @property
    def wqstd(self):
        return self._wqstd
    @wqstd.setter
    def wqstd(self, value):
        self._wqstd = value

    def wq_table(self, writeToFiles=True):
        """ Assembles a summary tables of WQ concentrations and loads

        Parameters
        ----------
        writeToFiles : bool, optional
            Determines if .csv and .tex representations of the output
            will be written.

        Writes
        ------
        Optionally writes .csv and .tex files of the table.

        Returns
        -------
        wqtable : pandas.DataFrame
            A concise DataFrame with water quality concetrations and loads.

        """

        if self.wqdata is not None:
            wqtable = (
                self.wqdata
                    .query("parameter != 'Escherichia coli'")
                    .merge(self.wqstd, on='parameter', suffixes=('', '_y'))
                    .rename(columns={'parameter': 'Parameter'})
            )

            wqtable['Effluent EMC'] = wqtable.apply(
                lambda r: self._res_with_units(r['concentration'], r['units']),
                axis=1
            )

            wqtable['Detection Limit'] = wqtable.apply(
                lambda r: self._res_with_units(r['detectionlimit'], r['units']),
                axis=1
            )

            wqtable['Effluent Load'] = wqtable.apply(
                lambda r: self._res_with_units(r['load_outflow'], r['load_units']),
                axis=1
            )

            wqtable['WQ Guideline'] = wqtable.apply(
                lambda r: self._res_with_units(r['upper_limit'], r['units']),
                axis=1
            )

            #wqtable = wqtable.rename(columns=lambda c: c.replace('_', ' ').title())
            cols_to_keep = [
                'Parameter',
                'WQ Guideline',
                'Detection Limit',
                'Effluent EMC',
                'Effluent Load'
            ]
            wqtable = wqtable[cols_to_keep].drop_duplicates()

            # pragma: no cover
            if writeToFiles:
                csvpath = os.path.join('output', 'csv', self.wq_tex_table + '.csv')
                texpath = os.path.join('output', 'tex', 'ISR', self.wq_tex_table + '.tex')

                wqtable.to_csv(csvpath, na_rep='--', index=False)
                utils.csvToTex(csvpath, texpath, pcols=25, replacestats=False)

            return wqtable

    @staticmethod
    def _write_basic_table(tablestring, filename):
        csv = os.path.join('output', 'csv', filename + '.csv')
        tex = os.path.join('output', 'tex', 'ISR', filename + '.tex')
        utils.makeTablesFromCSVStrings(tablestring, csvpath=csv)
        utils.csvToTex(csv, tex, pcols=0)
        with open(tex, 'r') as texfile:
            texstring = texfile.read()

        texstring = texstring.replace(r"\toprule", r"\midrule")
        texstring = texstring.replace(r"\bottomrule", r"\midrule")
        texstring = utils.sanitizeTex(texstring).replace(" nan ", " -- ")
        with open(tex, 'w') as texfile:
            texfile.write(texstring)

    def _make_ISR_tables(self):
        """ Creates tables (CSV and LaTeX) for the ISR reports. There are three
        tables currently included in ISRs:
            1) The general info table
            2) The hydrologic info table
            3) Water quality summary (composite data only)

        Parameters
        ----------
        None

        Writes
        ------
        LaTeX and CSV files of the tables list above and a water quality
        summary table of the data from grab samples.

        Returns
        -------
        None

        """

        ## get the string for Table 1
        # general = self.storm._general_table(self.tocentry)
        # self._write_basic_table(general, self.general_tex_table)

        # get and write the string for Table 2
        hydro = self.storm._hydro_table(self.tocentry)
        self._write_basic_table(hydro, self.hydro_tex_table)

        # make the WQ summary tables
        wq = self.wq_table()
        #return general, hydro, wq
        return hydro, wq

    def _make_ISR_document(self, version='draft'):
        """ Creates Individual Storm Report reports as LaTeX files.

        Parameters
        ----------
        version : string, optional (default = 'draft')
            Whether the file should be marked as "draft" or "final"

        Writes
        ------
        An Individual Storm Report LaTeX file the storm.

        Returns
        -------
        documentISR : str
            The file name and path of the ISR document.

        """

        # check input
        if version.lower() not in ('draft', 'final'):
            raise ValueError("Report version must be 'draft' or 'final")

        if version.lower() == 'draft':
            watermark = "\\usepackage{draftwatermark}\n\\SetWatermarkLightness{0.9}"
            draft = ' Draft'
        else:
            watermark = ''
            draft = ''

        # texpaths and placeholders in same order
        placeholders = {
            '__GeneralTable__': self.general_tex_table + '.tex',
            '__HydroTable__': self.hydro_tex_table + '.tex',
            '__WQTable__': self.wq_tex_table + '.tex',
            '__HydroFigure__': self.storm_figure + '.pdf',
            '__DATE__': self.starttime.strftime('%b %Y'),
            '__BMPPHOTO__': self.siteid,
            '__site_name__': self.tocentry,
            '__watermark__': watermark,
            '__draft__': draft
        }

        with open(self.templateISR, 'r') as f_template:
            template_string = f_template.read()

        for key in placeholders:
            template_string = template_string.replace(key, placeholders[key])

        documentISR_name = utils.processFilename('{}-{}-{}-ISR-{}.tex'.format(
            self.siteid, self.starttime.strftime('%Y%m%d-%H%M'),
            self.label.split(' ')[0], version
        ))
        documentISR = os.path.join('output', 'tex', 'ISR', documentISR_name)
        with open(documentISR, 'w') as f_report:
            f_report.write(template_string)

        return documentISR

    def make_samplefig(self, **figkwargs):
        """ Generate a matplotlib figure showing the hyetograph,
        hydrograph, and timing of water quality samples

        Parameters
        ----------
        figkwargs : keyward arguments
            Plotting options passed directly to Storm.summaryPlot

        Writes
        ------
        Saves a .png and .pdf of the figure

        Returns
        -------
        fig : matplotlib.figure
            The instance of the figure.

        """
        serieslabels = {
            self.storm.outflowcol: 'Effluent (L/s)',
            self.storm.precipcol: '10-min Precip Depth (mm)'
        }

        fig, artists, labels = self.storm.summaryPlot(inflow=False, showLegend=False,
                                                      figopts=figkwargs,
                                                      serieslabels=serieslabels)
        rug = self.plot_ts(ax=fig.axes[1], isFocus=True, asrug=False)
        fig.axes[0].set_ylabel('Precip (mm)')
        fig.axes[1].set_ylabel('BMP Effluent (L/s)')
        seaborn.despine(ax=fig.axes[1])
        seaborn.despine(ax=fig.axes[0], bottom=True, top=False)

        artists.extend([rug])
        labels.extend(['Samples'])

        leg = fig.axes[0].legend(artists, labels, fontsize=7, ncol=1,
                                 markerscale=0.75, frameon=False,
                                 loc='lower right')
        leg.get_frame().set_zorder(25)

        viz._savefig(fig, self.storm_figure, extra='Storm', asPDF=True)
        return fig

    def compileISR(self, version='draft', clean=False):
        """ Use ``pdflatex`` to compile the Individual Storm Report
         into a PDF document.

        Parameters
        ----------
        version : string, optional (default = 'draft')
            Whether the file should be marked as "draft" or "final"
        clean : bool, optional (default =  False)
            When True, all of the remnants of the LaTeX compilation
            process (e.g., .aux, .toc files) are removed.

        Writes
        ------
        A PDF document of the ISR

        Returns
        -------
        tex : string
            The filepath to the raw LaTeX file that was compiled into a
            PDF.

        Warning
        -------
        **Requires** ``pdflatex`` to be installed and visible on the
        system path.

        """

        # create the storm hyetograph/hydrograph
        fig = self.make_samplefig(figsize=(6, 3))
        plt.close(fig)

        # create the hydrologic and WQ summary tables
        hydro, wq = self._make_ISR_tables()

        # write the LaTeX source file
        texpath = self._make_ISR_document(version=version)

        texdir, texdoc = os.path.split(texpath)

        if wqio.testing.checkdep_tex() is not None:
            with LaTeXDirectory(texdir) as latex:
                tex = latex.compile(texdoc)

        else:
            tex = None

        return tex


class GrabSample(_WQSample_Mixin, wqio.GrabSample):
    pass


class CompositeSample(_WQSample_Mixin, wqio.CompositeSample):
    pass


class Storm(wqio.Storm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._siteid = None
        self._info = None

    @property
    def siteid(self):
        return self._siteid
    @siteid.setter
    def siteid(self, value):
        self._siteid = value

    @property
    def info(self):
        return self._info
    @info.setter
    def info(self, value):
        self._info = value

    @property
    def peak_lag_hours(self):
        if (self._peak_lag_hours is None and
                self.peak_outflow_time is not None and
                self.peak_precip_intensity_time is not None):

            time_delta = self.peak_outflow_time - self.peak_precip_intensity_time
            self._peak_lag_hours = time_delta.total_seconds() / wqio.hydro.SEC_PER_HOUR
        return self._peak_lag_hours

    @property
    def centroid_lag_hours(self):
        if (self._centroid_lag_hours is None and
                self.centroid_outflow_time is not None and
                self.centroid_precip_time is not None):
            self._centroid_lag_hours = (
                self.centroid_outflow_time - self.centroid_precip_time
            ).total_seconds() / wqio.hydro.SEC_PER_HOUR
        return self._centroid_lag_hours

    @property
    def lag(self):
        return self.centroid_lag_hours

    @np.deprecate
    def _general_table(self, name):
        """ Creates a simple string of a table of the basic storm info
        """

        return None

    def _hydro_table(self, name):
        """
        Creates a simple string of a table of the hydrologic results

        Parameters
        ----------
        name : string
            The name of the site as it should appear in the header
            of the table

        Writes
        ------
        None

        Returns
        -------
        table : string
            A CSV string of the general hydrologic results

        """

        if pandas.isnull(self.centroid_lag_hours):
            lagstring = '--'
        else:
            lagstring = '{:.1f}'.format(self.centroid_lag_hours * 60.)

        storm_values = self.info.copy()
        storm_values.update({
            'site': name,
            'eventdate': self.start.strftime('%Y-%m-%d %H:%M'),
            'drydays': self.antecedent_period_days
        })

        table = (
            "Site,{site:s}\n"
            "Event Date,{eventdate:s}\n"
            "Antecedent Dry Period,{drydays:.1f} days\n"
            "Event Duration,{duration_hours:.1f} hr\n"
            "Peak Effluent Flow,{peak_outflow:.1f} L/s\n"
            "Peak Precipitation Intensity,{peak_precip_intensity:.0f} mm/hr\n"
            "Lag Time,{centroid_lag_hours:.1f} hr\n"
            "Estimated Total Influent Volume,{inflow_m3:.0f} L\n"
            "Total Effluent Volume,{outflow_m3:.0f} L\n"
            "Total Precipitation,{total_precip_depth:.1f} mm\n"
        ).format(**storm_values)

        return table

    @np.deprecate
    def wideTableHeaders(self):
        wide_header = (
            '"Storm Date",'
            '"Sample Date",'
            '"Antecedent Dry Period (days)",'
            '"Event Duration (hr)",'
            '"Peak Precipitation Intensity (mm/hr)",'
            '"Total Precipitation (mm)",'
            '"Peak Effluent Flow (L/s)",'
            '"Total Estimated Influent Volume (L)",'
            '"Total Effluent Volume (L)",'
            '"Centroid Lag Time (hr)",'
            '"Estimated Runoff Volume Reduction (L)",'
            '"Estimated Runoff Volume Reduction (%)"\n'
        )
        return wide_header

    @np.deprecate
    def wideTableLine(self):
        """ Creates a line to the "wide" table summarizing all storms
        for a site.

        Parameters
        ----------
        None

        Writes
        ------
        None

        Returns
        -------
        txt : string
            CSV string for a single row in the larger summary table

        """

        txt = '"{0:%Y-%m-%d}","{1}",{2:0.1f},"{3:,.0f}",' \
              '{4:0.1f},{5:0.1f},{6:0.1f},'\
              '"{7:,.0f}","{8:,.0f}","{9:,.0f}",' \
              '"{10:,.0f}","{11}%"\n'.format(
                self.storm_start, utils.stringify(self.sampledate, '%s'),
                self.antecedent_duration, self.duration,
                self.peak_intensity, self.total_precip,
                self.peak_flow, self.influent_volume,
                self.total_volume, self.lag,
                self.volume_reduction_liters,
                utils.stringify(self.volume_reduction_percent, '%d')
              )
        return txt

    @np.deprecate
    def thinTableHeaders(self):
        thin_header = (
            '"Storm Date",'
            '"Total Precipitation (mm)",'
            '"Lag Time (min)",'
            '"Total Estimated Influent Volume (L)",'
            '"Total Effluent Volume (L)",'
            '"Estimated Peak Runoff (L/s)",'
            '"Peak Effluent Flow (L/s)"\n'
        )
        return thin_header

    @np.deprecate
    def thinTableLine(self):
        """ Creates a line to the "thin" table summarizing all storms
        for a site.

        Parameters
        ----------
        None

        Writes
        ------
        None

        Returns
        -------
        txt : string
            CSV string for a single row in the larger summary table

        """

        txt = '%s,%0.1f,%0.0f,%0.0f,%0.0f,%0.1f,%0.1f' % (
            self.storm_start, self.total_precip, self.lag,
            self.influent_volume, self.total_volume, self.peak_flow,
            self.peak_influent
        )
        return txt
