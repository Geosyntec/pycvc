import os
import sys
import glob
from pkg_resources import resource_filename

import numpy as np
from scipy import interpolate
import pandas
import pyodbc
import seaborn.apionly as seaborn

import wqio
from wqio import utils

# CVC project info
from . import info
from . import viz

# CVC-specific wqio.events subclasses
from .samples import GrabSample, CompositeSample, Storm


__all__ = ['Database', 'Site']


LITERS_PER_CUBICMETER = 1000.
MILLIMETERS_PER_METER = 1000.


def _check_sampletype(sampletype):
    """ Confirms that a given value is a valid sampletype and returns
    the all lowercase version of it.
    """
    if sampletype.lower() not in ('grab', 'composite'):
        raise ValueError("`sampletype` must be 'composite' or 'grab'")

    return sampletype.lower()


def _check_rescol(rescol):
    """ Comfirms that a give value is a valid results column and returns
    the corresponding units column and results column.
    """
    if rescol.lower() == 'concentration':
        unitscol = 'units'
    elif rescol.lower() == 'load_outflow':
        unitscol = 'load_units'
    else:
        raise ValueError("`rescol` must be in ['concentration', 'load_outflow']")
    return rescol.lower(), unitscol


def _check_timegroup(timegroup):
    valid_groups = ['season', 'grouped_season', 'year']
    if timegroup is None:
        return timegroup
    elif timegroup.lower() in valid_groups:
        return timegroup.lower()
    else:
        raise ValueError("{} is not a valid time group ({})".format(timegroup, valid_groups))


def _grouped_seasons(timestamp):
    season = utils.getSeason(timestamp)
    if season.lower() in ['spring', 'winter']:
        return 'winter/spring'
    elif season.lower() in ['summer', 'autumn']:
        return 'summer/autumn'
    else:
        raise ValueError("{} is not a valid season".format(season))


def _remove_storms_from_df(df, dates, datecol):
    # determine which storm numbers should be excluded
    excluded_storms = []
    if dates is not None:
        # stuff in a list if necessary
        if np.isscalar(dates):
            dates = [dates]

        # loop through all of the excluded dates
        for d in dates:
            # # convert to a proper python date object
            excl_date = wqio.utils.santizeTimestamp(d).date()
            storm_rows = df.loc[df[datecol].dt.date == excl_date]
            excluded_storms.extend(storm_rows['storm_number'].values)

    return df.query("storm_number not in @excluded_storms")


class Database(object):
    """ Class representing the CVC database, providing quick access
    to site-specific data and information.

    Parameters
    ----------
    dbfile : string
        Path to the CVC Microsoft Access database (.accdb format)
    nsqddata : cvc.external.nsqd object
        Data structure representing the National Stormwater Quality
        Dataset.
    bmpdb : cvc.external.nsqd object
        Data structure representing the Internation Stormwater BMP
        Database.
    testing : bool (default = False)
        When True, the data only go back to 2014 to speed up the
        analysis.

    """

    def __init__(self, dbfile, nsqdata=None, bmpdb=None, testing=False):
        self.dbfile = dbfile
        self.nsqdata = nsqdata
        self.bmpdb = bmpdb
        self.testing = testing
        self._sites = None
        self._wqstd = None

    def connect(self):
        """ Connects to the database.

        Parameters
        ----------
        None

        Returns
        -------
        cnn : pyodbc database connection object

        """

        driver = r'{Microsoft Access Driver (*.mdb, *.accdb)}'
        connection_string = r'Driver=%s;DBQ=%s' % (driver, self.dbfile)
        return pyodbc.connect(connection_string)

    @property
    def sites(self):
        if self._sites is None:
            with self.connect() as cnn:
                self._sites = pandas.read_sql("select * from sites", cnn)
        return self._sites

    def _check_site(self, site):
        """ Validates a string as a site that exists in the database.

        """

        if not site in self.sites['site'].tolist():
            raise ValueError('invalid site')
        else:
            return site

    def _run_query(self, qry):
        """ Executes a query against the database

        Parameters
        ----------
        qry : string
            A valid Microsoft Access query statement.

        Returns
        -------
        df : pandas.DataFrame

        """

        with self.connect() as cnn:
            df = pandas.read_sql(qry, cnn)

        return df

    def getWQData(self, site, onlyPOCs=True):
        """ Fetches water quality data for a CVC site.

        Parameters
        ----------
        site : string
            The name of the CVC site whose data should be returned.
        onlyPOCs : bool, optional (default = True)
            When True, the data are limited to just the main pollutants
            of concern (e.g., nutrients, metals, TSS).

        Returns
        -------
        wq : pandas.DataFrame

        """

        site = self._check_site(site)

        qry = """
        select
            s.site,
            s.sample as [sample],
            s.sampletype,
            s.samplestart,
            s.samplestop,
            s.interval_minutes,
            r.parameter,
            r.units,
            r.detectionlimit,
            r.qualifier,
            r.concentration
        from results r
        inner join samples s
            on s.sample = r.sample
        where
            s.site = '{}'
        and s.labtype = 'REGULAR'
        and s.ignore = No
        """.format(site)

        wq = self._run_query(qry)
        if onlyPOCs:
            wq = wq[wq['parameter'].isin(info.getPOCs())]

        if self.testing:
            wq = wq[wq['samplestart'].dt.year == 2014]

        return wq

    def getHydroData(self, site, resamplePeriodMinutes=10):
        """ Fetches hydrologic data for a CVC site.

        Parameters
        ----------
        site : string
            The name of the CVC site whose data should be returned.
        resamplePeriodMinutes : int, optional (default = 10)
            Data provided from CVC originally came varying observations
            frequencies. The parameter resamples the data to a
            consistent period.

        Returns
        -------
        wq : pandas.DataFrame

        """

        site = self._check_site(site)

        qry = """
        select
            hydro.datetime,
            hydro.level_mm - sites.weir_height_mm as [head_mm],
            hydro.precip_mm,
            hydro.flow_lps as _raw_flow_lps
        from
            hydro
        inner join
            sites on sites.site = hydro.site
        where
            sites.site = '{}'
        """.format(site)

        hydro = self._run_query(qry).set_index('datetime')
        if resamplePeriodMinutes is not None:
            resample_dict = {
                'head_mm': 'mean',
                'precip_mm': 'sum',
            }
            resampleOffset = pandas.offsets.Minute(resamplePeriodMinutes)
            hydro = hydro.resample(resampleOffset, how=resample_dict)

        if self.testing:
            hydro = hydro.loc['2014']

        return hydro

    def getDrainageArea(self, site):
        """ Fetches draiange area information for a CVC site.

        Parameters
        ----------
        site : string
            The name of the CVC site whose data should be returned.

        Returns
        -------
        da : wqio.DrainageArea

        """

        site = self._check_site(site)

        site_info = self.sites[self.sites['site'] == site]
        da = wqio.DrainageArea(
            total_area=site_info['total_area'],
            imp_area=site_info['impervious_area'],
            bmp_area=site_info['bmp_area']
        )

        return da

    def getSamples(self, site):
        """ Fetches information for non-QA, non-ignored samples
        collected at a CVC site.

        Parameters
        ----------
        site : string
            The name of the CVC site whose data should be returned.

        Returns
        -------
        samples : pandas.DataFrame

        """

        site = self._check_site(site)
        qry = """
        select *
        from samples
        where site = '{}'
          and labtype = 'REGULAR'
          and ignore = No
        """.format(site)

        samples = self._run_query(qry)

        if self.testing:
            samples = samples[samples['samplestart'].dt.year == 2014]

        return samples

    def getRatingCurve(self,  site):
        """ Fetches rating curve information for a CVC site.

        Parameters
        ----------
        site : string
            The name of the CVC site whose data should be returned.

        Returns
        -------
        rating_curve : pandas.DataFrame

        """

        site = self._check_site(site)
        qry = """
        select head_mm, flow_lps
        from rating_curves
        where site = '{}'
        """.format(site)
        return self._run_query(qry)

    @property
    def wqstd(self):
        """ WQ guidelines provided by CVC.
        """
        if self._wqstd is None:
            self._wqstd = self._run_query("select * from wq_standards")
            joincols = ['parameter', 'units']
            if self.nsqdata is not None:
                self._wqstd = self._wqstd.merge(self.nsqdata.medians, on=joincols)

            if self.bmpdb is not None:
                self._wqstd = self._wqstd.merge(self.bmpdb.medians, on=joincols)

        return self._wqstd


class Site(object):
    """ Representation of a CVC site in data form with various utilities
    to quickly summarize that data.

    Parameters
    ----------
    db : cvc.Database object
        The object representing the database.
    siteid : string
        The ID of the site whose water quality and hydrologic data will
        be analyzed (e.g., ED-1, LV-4).
    raingauge : string
        The ID of the rain gauge to use (flow data still comes from
        ``siteid``).
    onlyPOCs : bool, optional (default = True)
        When True, the data are limited to just the main pollutants
        of concern (e.g., nutrients, metals, TSS).
    tocentry : string or None (default)
        Full-text representation of the site's name as it should appear
        in a table of contents (TOC) or report heading.
    hydroPeriodMinutes : int, optional (default = 10)
        Data provided from CVC originally came varying observations
        frequencies. The parameter resamples the data to a
        consistent period.
    intereventHours : float or int, optional (default = 6)
        The minimum dry-duration (no flow or rain) that delineates two
        distinct storms.
    compperiods : int, optional (default = 18)
        The number of aliquots collected during a composite sample.
    isreference : bool, optional (default = False)
        Flags a site a being a reference/control site. In other words,
        only set to True for LV-1.
    minflow, minprecip : float, optional (defaults = 0.0)
        The minimum amount of flow or precip required for a hydrologic
        event to be considered a proper storm.
    influentmedians : pandas.DataFrame
        A rough approximation of the central tendency of the influent
        water quality.
    runoff_fxn, bypass_fxn, inflow_fxn : callable
        Functions that compute their various water balance elements
        from the other measured hydrologic quantities.
    color, marker : string, optional
        A valid matplotlib symbology values.

    """

    def __repr__(self):
        return self.siteid

    def __init__(self, db, siteid, raingauge, onlyPOCs=True, tocentry=None,
                 hydroPeriodMinutes=10, intereventHours=6, compperiods=18,
                 isreference=False, minflow=0.0, minprecip=0.0,
                 influentmedians=None, templateISR=None,
                 runoff_fxn=None, bypass_fxn=None, inflow_fxn=None,
                 color='blue', marker='o'):

        # main args/kwargs
        self.db = db
        self.siteid = siteid
        self.raingauge = raingauge
        self.hydroPeriodMinutes = hydroPeriodMinutes
        self.intereventHours = intereventHours
        self.compperiods = compperiods
        self.minflow = minflow
        self.minprecip = minprecip
        self._influentmedians = influentmedians
        self.isreference = isreference
        self.onlyPOCs = onlyPOCs
        self.runoff_fxn = (lambda x: np.nan) if runoff_fxn is None else runoff_fxn
        self.bypass_fxn = (lambda x: np.nan) if bypass_fxn is None else bypass_fxn
        self.inflow_fxn = (lambda x: np.nan) if inflow_fxn is None else inflow_fxn
        self.tocentry = tocentry
        self.color = color
        self.marker = marker

        # properties
        if templateISR is None:
            templateISR = resource_filename("pycvc.tex", "template_isr.tex")
        self._templateISR = templateISR
        self._drainagearea = None
        self._wqstd = None
        self._wqdata = None
        self._sample = None
        self.__rating_curve_data = None
        self._rating_curve = None
        self._hydrodata = None
        self._grabdates = None
        self._compdates = None
        self._compendtimes = None
        self._max_precip = None
        self._max_flow = None
        self._max_inflow = None
        self._sampled_storms = None
        self._unsampled_storms = None
        self._storms = None
        self._storm_info = None
        self._tidy_data = None
        self._sample_info = None
        self._all_samples = None
        self._samples = None

    @property
    def influentmedians(self):
        return self._influentmedians
    @influentmedians.setter
    def influentmedians(self, value):
        self._influentmedians = value

    @property
    def runoff_fxn(self):
        return self._runoff_fxn
    @runoff_fxn.setter
    def runoff_fxn(self, value):
        self._runoff_fxn = value

    @property
    def bypass_fxn(self):
        return self._bypass_fxn
    @bypass_fxn.setter
    def bypass_fxn(self, value):
        self._bypass_fxn = value

    @property
    def inflow_fxn(self):
        return self._inflow_fxn
    @inflow_fxn.setter
    def inflow_fxn(self, value):
        self._inflow_fxn = value

    @property
    def wqstd(self):
        """ Water quality guidelines/standards """
        if self._wqstd is None:
            self._wqstd = self.db.wqstd.copy()
        return self._wqstd

    @property
    def _rating_curve_data(self):
        """ Raw rating curve at a site """
        # load the rating curve
        if self.__rating_curve_data is None:
            self.__rating_curve_data = self.db.getRatingCurve(self.siteid)
        return self.__rating_curve_data

    @property
    def rating_curve(self):
        """ Interpolation function that converts head above the weir
        (mm) to a flow rate (liters/second) """
        # load the rating curve
        if self._rating_curve is None:
            self._rating_curve = interpolate.interp1d(
                self._rating_curve_data['head_mm'],
                self._rating_curve_data['flow_lps'],
                bounds_error=False
            )
        return self._rating_curve

    @property
    def drainagearea(self):
        """ wqio.DrainageArea for the site """
        if self._drainagearea is None:
            self._drainagearea = self.db.getDrainageArea(self.siteid)
        return self._drainagearea

    @property
    def wqdata(self):
        if self._wqdata is None:
            self._wqdata = self.db.getWQData(
                self.siteid, onlyPOCs=self.onlyPOCs
            )
            self._wqdata = (
                self._wqdata
                    .assign(season=self._wqdata['samplestart'].apply(utils.getSeason))
                    .assign(grouped_season=self._wqdata['samplestart'].apply(_grouped_seasons))
                    .assign(year=self._wqdata['samplestart'].dt.year.astype(str))
                    .assign(sampletype=self._wqdata['sampletype'].str.lower())
            )
        return self._wqdata

    @property
    def hydrodata(self):
        """ wqio.HydroRecord for the site. """
        if self._hydrodata is None:
            head = self.db.getHydroData(
                self.siteid, resamplePeriodMinutes=self.hydroPeriodMinutes
            )[['head_mm']]

            precip = self.db.getHydroData(
                self.raingauge, resamplePeriodMinutes=self.hydroPeriodMinutes
            )[['precip_mm']]

            hydro = precip.join(head, how='outer')
            hydro['flow_lps'] = self.rating_curve(hydro['head_mm'])

            ## rev: used to apply the simple method to everytime
            ##      timestep of the hydrodata to estimated inflow.
            ##      that was a really bad idea and now we just use
            ##      the regression equations to estimate the total
            ##      inflow volume for entire storms.
            # hydro['inflow_lps'] = self.drainagearea.simple_method(
            #     hydro['precip_mm']
            # ) / 60. * self.hydroPeriodMinutes
            hydro['inflow_lps'] = np.nan

            self._hydrodata = wqio.HydroRecord(
                hydro, precipcol='precip_mm', outflowcol='flow_lps',
                inflowcol='inflow_lps', stormcol='storm',
                minprecip=self.minprecip, mininflow=self.minflow,
                minoutflow=self.minflow, outputfreqMinutes=10,
                intereventHours=self.intereventHours,
                volume_conversion=1./LITERS_PER_CUBICMETER,
                stormclass=Storm, lowmem=True
            )

        return self._hydrodata

    @property
    def sample_info(self):
        """ DataFrame of sample information at the site """
        if self._sample_info is None:
            self._sample_info = self.db.getSamples(self.siteid)
            for col in ['sampletype', 'labtype']:
                self._sample_info[col] = self._sample_info[col].str.lower()
        return self._sample_info

    @property
    def grabdates(self):
        """ Dates and times at which grab samples were collected """
        if self._grabdates is None:
            self._grabdates = self._get_sample_dates('grab', which='start')
        return self._grabdates

    @property
    def compdates(self):
        """ Dates and times at which composite samples were initiated """
        if self._compdates is None:
            self._compdates = self._get_sample_dates('composite', which='start')
        return self._compdates

    @property
    def compendtimes(self):
        """ Dates and times at which composite samples were finshed """
        if self._compendtimes is None:
            self._compendtimes = self._get_sample_dates('composite', which='end')
        return self._compendtimes

    @property
    def max_precip(self):
        """ Max precip *depth* in the hydrologic record for the site"""
        if self._max_precip is None:
            self._max_precip = self.hydrodata.data['precip_mm'].max()
        return self._max_precip

    @property
    def max_flow(self):
        """ Max outflow in the hydrologic record for the site"""
        if self._max_flow is None:
            self._max_flow = self.hydrodata.data['flow_lps'].max()
        return self._max_flow

    @property
    def max_inflow(self):
        """ Max outflow in the hydrologic record for the site"""
        if self._max_inflow is None:
            self._max_inflow = self.hydrodata.data['inflow_lps'].max()
        return self._max_inflow

    @property
    def sampled_storms(self):
        """ Storms during which a water quality sample was collected """
        if self._sampled_storms is None:
            self._sampled_storms = self._get_storms_with_data(sampletype='composite')
            self._sampled_storms.sort()
        return self._sampled_storms

    @property
    def unsampled_storms(self):
        """ Storms during which a water quality sample was not collected """
        if self._unsampled_storms is None:
            self._unsampled_storms = self._get_storms_without_data(sampletype='composite')
            self._unsampled_storms.sort()
        return self._unsampled_storms

    @property
    def storms(self):
        """ All the storms observed at a site """
        if self._storms is None:
            for s in self.hydrodata.storms.keys():
                st_info = self.storm_info[self.storm_info['storm_number'] == s]
                self.hydrodata.storms[s].info = st_info.iloc[0].to_dict()
            self._storms = self.hydrodata.storms
        return self._storms

    @property
    def all_samples(self):
        """ All the water quality samples collected at a site """
        if self._all_samples is None:
            self._all_samples = []
            for samplename in self.sample_info['sample'].unique():
                sample = self._get_sample_from_name(samplename)
                self._all_samples.append(sample)
        return self._all_samples

    @property
    def samples(self):
        """ Dictionary of 'grab' and 'composite' water quality samples """
        if self._samples is None:
            self._samples = {
                'grab': list(filter(lambda s: isinstance(s, GrabSample), self.all_samples)),
                'composite': list(filter(lambda s: isinstance(s, CompositeSample), self.all_samples))
            }
        return self._samples

    @property
    def storm_info(self):
        """ DataFrame summarizing each storm event """
        intensity_to_depth = self.hydroPeriodMinutes / wqio.hydro.MIN_PER_HOUR
        volume_to_flow_liters = 1.0 / (wqio.hydro.SEC_PER_MINUTE * self.hydroPeriodMinutes)
        full_area = self.drainagearea.total_area + self.drainagearea.bmp_area

        def fake_peak_inflow(row):
            precip = row['peak_precip_intensity']
            return self.drainagearea.simple_method(precip * intensity_to_depth) * volume_to_flow_liters

        if self._storm_info is None:
            stormstats = (
                self.hydrodata
                    .storm_stats
                    .rename(columns=lambda c: c.lower().replace(' ', '_'))
                    .assign(site=self.siteid)
                    .assign(grouped_season=lambda df: df['start_date'].apply(_grouped_seasons))
                    .assign(year=lambda df: df['start_date'].dt.year.astype(str))
                    .fillna(value={'Total Precip Depth': 0})
                    .rename(columns={'total_outflow_volume': 'outflow_m3'})
                    .fillna(value={'outflow_m3': 0})
                    .assign(sm_est_peak_inflow=lambda df: df.apply(fake_peak_inflow, axis=1))
                    .assign(outflow_mm=lambda df: df['outflow_m3'] / full_area * MILLIMETERS_PER_METER)
                    .assign(runoff_m3=lambda df: df.apply(self.runoff_fxn, axis=1))
                    .assign(bypass_m3=lambda df: df.apply(self.bypass_fxn, axis=1))
                    .assign(inflow_m3=lambda df: df.apply(self.inflow_fxn, axis=1))
                    .assign(has_outflow=lambda df: df['outflow_m3'].apply(lambda r: 'Yes' if r > 0.1 else 'No'))
            )

            col_order = [
                'site', 'storm_number','year', 'season', 'grouped_season',
                'antecedent_days', 'start_date', 'end_date', 'duration_hours',
                'peak_precip_intensity', 'total_precip_depth',
                'runoff_m3', 'bypass_m3', 'inflow_m3', 'outflow_m3', 'outflow_mm',
                'peak_outflow', 'centroid_lag_hours', 'peak_lag_hours',
                'has_outflow', 'sm_est_peak_inflow'
            ]

            self._storm_info = stormstats[col_order]

        return self._storm_info

    def storm_stats(self, minprecip=0, excluded_dates=None, timegroup=None, **winsor_params):
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
        timegroup : string, optional
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

        timecol = _check_timegroup(timegroup)
        if timecol is None:
            groups = ['site']
        else:
            groups = ['site', timecol]

        data = (
            self.storm_info
                .pipe(utils.winsorize_dataframe, **winsor_params)
                .pipe(_remove_storms_from_df, excluded_dates, 'start_date')
                .query("total_precip_depth > @minprecip")
        )

        descr = data.groupby(by=groups).describe()
        descr.index.names = groups + ['stat']
        descr = descr.select(lambda c: c != 'storm_number', axis=1)
        descr.columns.names = ['quantity']
        storm_stats = (
            descr.stack(level='quantity')
                 .unstack(level='stat')
                 .xs(self.siteid, level='site')
        )
        return storm_stats

    @property
    def tidy_data(self):
        """ A concise DataFrame with the relevant water quality and
        hydrologic data. See: http://www.jstatsoft.org/v59/i10/paper """

        final_cols = [
            'site', 'sample', 'sampletype', 'season', 'grouped_season', 'year',
            'samplestart', 'samplestop', 'interval_minutes', 'parameter', 'units',
            'detectionlimit', 'qualifier', 'concentration',
            'influent lower', 'influent median', 'influent upper',
            'storm_number', 'antecedent_days', 'start_date', 'end_date',
            'duration_hours', 'peak_precip_intensity_mm_per_hr', 'total_precip_depth_mm',
            'runoff_m3', 'bypass_m3', 'inflow_m3', 'outflow_m3',
            'peak_outflow_L_per_s', 'centroid_lag_hours', 'peak_lag_hours',
            'load_runoff_lower', 'load_runoff', 'load_runoff_upper',
            'load_bypass_lower', 'load_bypass', 'load_bypass_upper',
            'load_inflow_lower', 'load_inflow', 'load_inflow_upper',
            'load_outflow', 'load_units', 'load_factor',
        ]

        if self._tidy_data is None:
            td = self.wqdata.copy()
            rename_cols = {
                'peak_precip_intensity': 'peak_precip_intensity_mm_per_hr',
                'total_precip_depth': 'total_precip_depth_mm',
                'peak_outflow': 'peak_outflow_L_per_s'
            }

            storm_merge_cols = ['site', 'storm_number', 'year', 'season', 'grouped_season']

            # add load and storm information
            td = (
                td.assign(sampletype=td['sampletype'].str.lower())
                  .assign(storm_number=td['samplestart'].apply(lambda d: self._get_storm_from_date(d)[0]))
                  .assign(load_units=td['parameter'].apply(lambda p: info.getPOCInfo('cvcname', p, 'load_units')))
                  .assign(load_factor=td['parameter'].apply(lambda p: info.getPOCInfo('cvcname', p, 'load_factor')))
                  .merge(self.storm_info, on=storm_merge_cols, how='right')
                  .rename(columns=rename_cols)
            )

            # join in the influent medians if they exist
            influent_merge_cols = ['parameter', 'season', 'units']
            if self.influentmedians is not None:
                td = td.merge(self.influentmedians, on=influent_merge_cols, how='outer')
            else:
                td['influent median'] = np.nan
                td['influent lower'] = np.nan
                td['influent upper'] = np.nan

            # compute loads
            self._tidy_data = (
                td.assign(load_runoff_lower=td['influent lower'] * td['runoff_m3'] * td['load_factor'])
                  .assign(load_runoff=td['influent median'] * td['runoff_m3'] * td['load_factor'])
                  .assign(load_runoff_upper=td['influent upper'] * td['runoff_m3'] * td['load_factor'])
                  .assign(load_bypass_lower=td['influent lower'] * td['bypass_m3'] * td['load_factor'])
                  .assign(load_bypass=td['influent median'] * td['bypass_m3'] * td['load_factor'])
                  .assign(load_bypass_upper=td['influent upper'] * td['bypass_m3'] * td['load_factor'])
                  .assign(load_inflow_lower=td['influent lower'] * td['inflow_m3'] * td['load_factor'])
                  .assign(load_inflow=td['influent median'] * td['inflow_m3'] * td['load_factor'])
                  .assign(load_inflow_upper=td['influent upper'] * td['inflow_m3'] * td['load_factor'])
                  .assign(load_outflow=td['concentration'] * td['outflow_m3'] * td['load_factor'])
                  .dropna(subset=['site'])
            )[final_cols]

        return self._tidy_data

    @property
    def templateISR(self):
        """ LaTeX template for generating Individual Storm Reports """
        return self._templateISR
    @templateISR.setter
    def templateISR(self, value):
        self._templateISR = value

    def _get_sample_from_name(self, samplename):
        """ Returns a wqio.GrabSample or wqio.CompositeSample from a
        samplename.
        """
        sampleinfo = self.sample_info.query("sample == @samplename").iloc[0]
        wqdata = self.tidy_data.query("sample == @samplename")
        if sampleinfo['sampletype'] == 'grab':
            SampleObject = GrabSample
        elif sampleinfo['sampletype'] == 'composite':
            SampleObject = CompositeSample
        else:
            raise ValueError("invalid sampletype {} for {}".format(
                sampleinfo['sampletype'], samplename
            ))

        storm_number, storm = self._get_storm_from_date(sampleinfo['samplestart'])
        #storms = list(filter(lambda s: s['number'] == storm_number, self.storms))

        sample = SampleObject(
            wqdata, sampleinfo['samplestart'],
            samplefreq=pandas.offsets.Minute(sampleinfo['interval_minutes']),
            endtime=sampleinfo['samplestop'],
            rescol='concentration',
            qualcol='qualifier',
            dlcol='detectionlimit',
            unitscol='units',
            storm=storm
        )
        sample.siteid = self.siteid
        sample.tocentry = self.tocentry
        sample.wqstd = self.wqstd.copy()
        sample.templateISR = self.templateISR
        return sample

    def _get_sample_dates(self, sampletype, which='start'):
        """ Returns of the sample collected start dates/times for a
        gived sampletype
        """
        sampletype = _check_sampletype(sampletype)

        if which.lower() == 'start':
            finalcol = 'samplestart'
        elif which.lower() in ('end', 'stop'):
            finalcol = 'samplestop'
        else:
            raise ValueError("'which' must be one of ('start', 'end', 'stop')")

        subset = self.sample_info.query("sampletype == @sampletype")
        return subset[finalcol]

    def _get_storms_with_data(self, sampletype='composite'):
        """ Returns all of the storm numbers of storms where a given
        sampletype was collected.
        """
        sampletype = _check_sampletype(sampletype)
        return self.tidy_data.query("sampletype == @sampletype")['storm_number'].unique()

    def _get_storms_without_data(self, sampletype='composite'):
        """ Returns all of the storm numbers of storms where no water
        quality samples was collected.
        """
        sampled = self._get_storms_with_data(sampletype=sampletype)
        return np.array(list(filter(lambda sn: sn not in sampled, self.storms.keys())))

    def _get_storm_from_date(self, sampledate):
        """ Looks up a storm number from a timestamp """
        storm_number, storm = self.hydrodata.getStormFromTimestamp(
            sampledate,
            lookback_hours=self.intereventHours
        )
        return storm_number, storm

    def _get_dates_from_storm(self, storm_number, sampletype='grab'):
        """ Looks up the startdate of a sample from a storm number
        and sample type.
        """

        # pick out the right dates/validate input
        if sampletype == 'grab':
            dates = self.grabdates.tolist()

        elif sampletype == 'comp':
            dates = self.compdates.tolist()

        elif sampletype == 'any':
            dates = self.grabdates.tolist()
            dates.extend(self.compdates.tolist())

        else:
            raise ValueError("only 'grab', 'comp', or 'any' samples available")

        # go through and list all of the sample dates in that storm
        sampledates = []
        for d in dates:
            sn, storm = self._get_storm_from_date(d)
            if storm_number is not None and sn == storm_number:
                sampledates.append(d)

        if len(sampledates) == 0:
            return None

        return sampledates

    def _wq_summary(self, rescol='concentration', sampletype='composite', excluded_dates=None, timegroup=None):
        """ Returns a dataframe of seasonal or overall water quality
        stats for the given sampletype.
        """
        rescol, unitscol = _check_rescol(rescol)
        if timegroup is None:
            groupcols = ['parameter', unitscol]
        else:
            groupcols = ['parameter', unitscol, timegroup]

        data = _remove_storms_from_df(self.tidy_data, excluded_dates, "start_date")

        summary_percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        all_data = (
            data.query("sampletype == @sampletype")
                .groupby(by=groupcols)[rescol]
                .apply(lambda g: g.describe(percentiles=summary_percentiles))
                .unstack(level=-1)
        )

        if self.tidy_data.query("sampletype == @sampletype and qualifier != '='").shape[0] > 0:
            nd_data = (
                data.query("sampletype == @sampletype and qualifier != '='")
                    .groupby(by=groupcols)[rescol]
                    .size()
                    .to_frame()
                    .rename(columns={0: 'count NDs'})
            )

            all_data = all_data.join(nd_data).fillna({'count NDs': 0})
        else:
            all_data['count NDs'] = 0

        all_data['cov'] = all_data['std'] / all_data['mean']

        columns = (
            all_data.columns[:1].tolist() + nd_data.columns.tolist() +
            all_data.columns[1:3].tolist() + all_data.columns[-1:].tolist() +
            all_data.columns[3:-1].tolist()
        )
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

        return all_data[columns].rename(columns=stat_labels)

    def wq_summary(self, rescol='concentration', sampletype='composite', excluded_dates=None, timegroup=None):
        """ Basical water quality Statistics

        Parameters
        ----------
        rescol : string (default = 'concentration')
            The result column to summaryize. Valid values are
            "concentration" and "load_outflow".
        sampletype : string (default = 'composite')
            The types of samples to be summarized. Valid values are
            "composite" and "grab".
        excluded_dates : list of date-likes, optional
            This is a list of storm start dates that will be removed
            from the storms dataframe prior to computing statistics.
        timegroup : string, optional
            Optional string that defined how results should be group
            temporally. Valid options are "season", "grouped_season",
            and year. Default behavior does no temporal grouping.

        Returns
        -------
        summary : pandas.DataFrame

        """

        sampletype = _check_sampletype(sampletype)
        rescol, unitscol = _check_rescol(rescol)
        timecol = _check_timegroup(timegroup)

        overall = self._wq_summary(rescol=rescol, sampletype=sampletype,
                                   excluded_dates=excluded_dates, timegroup=None)
        if timecol is None:
            return overall
        else:
            seasonal = self._wq_summary(rescol=rescol, sampletype=sampletype,
                                        excluded_dates=excluded_dates, timegroup=timegroup)
            overall[timecol] = 'all'
            overall = overall.reset_index().set_index(['parameter', unitscol, timecol])
            return seasonal.append(overall).sort_index().T

    def medians(self, rescol='concentration', sampletype='composite', timegroup=None):
        """ Returns a DataFrame of the WQ medians for the given
        sampletype.

        Parameters
        ----------
        rescol : string (default = 'concentration')
            The result column to summaryize. Valid values are
            "concentration" and "load_outflow".
        sampletype : string (default = 'composite')
            The types of samples to be summarized. Valid values are
            "composite" and "grab".
        timegroup : string, optional
            Optional string that defined how results should be group
            temporally. Valid options are "season", "grouped_season",
            and year. Default behavior does no temporal grouping.

        Returns
        -------
        medians : pandas.DataFrame

        """

        sampletype = _check_sampletype(sampletype)
        rescol, unitscol = _check_rescol(rescol)
        timecol = _check_timegroup(timegroup)

        if timecol is None:
            othergroups = [unitscol]
        else:
            othergroups = [unitscol, timecol]

        dc_options = dict(rescol=rescol, qualcol='qualifier', ndval='<',
                          othergroups=othergroups, stationcol='site')
        medians = (
            self.tidy_data
                .query("sampletype == @sampletype")
                .reset_index()
                .pipe(wqio.DataCollection, **dc_options)
                .medians[self.siteid]
                .rename(columns={'stat': 'median'})
                .rename(columns=lambda c: 'effluent {}'.format(c))
                .reset_index()
        )
        return medians

    def sampled_loads(self, sampletype='composite', excluded_dates=None, timegroup=None, NAval=None):
        """ Returns the total loads for sampled storms and the given
        sampletype.

        Parameters
        ----------
        sampletype : string (default = 'composite')
            The types of samples to be summarized. Valid values are
            "composite" and "grab".
        timegroup : string, optional
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

        sampletype = _check_sampletype(sampletype)
        timecol = _check_timegroup(timegroup)
        if timecol is None:
            groupcols = ['parameter', 'load_units']
        else:
            groupcols = ['parameter', timecol, 'load_units']

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
            'load_outflow': 'sum',
        }
        loads = (
            self.tidy_data
                .query("sampletype == @sampletype")
                .pipe(_remove_storms_from_df, excluded_dates, "start_date")
                .groupby(by=groupcols)
                .agg(agg_dict)
        )

        if NAval is not None:
            loads = loads.fillna(NAval)

        def total_reduction(df, incol, outcol):
            return df[incol] - df[outcol]

        def pct_reduction(df, redcol, incol):
            return df[redcol] / df[incol] * 100.0

        loads = (
           loads.assign(reduct_mass_lower=lambda df: total_reduction(df, 'load_inflow_lower', 'load_outflow'))
                .assign(reduct_pct_lower=lambda df: total_reduction(df, 'reduct_mass_lower', 'load_inflow_lower'))
                .assign(reduct_mass=lambda df: total_reduction(df, 'load_inflow', 'load_outflow'))
                .assign(reduct_pct=lambda df: total_reduction(df, 'reduct_mass', 'load_inflow'))
                .assign(reduct_mass_upper=lambda df: total_reduction(df, 'load_inflow_upper', 'load_outflow'))
                .assign(reduct_pct_upper=lambda df: total_reduction(df, 'reduct_mass_upper', 'reduct_mass_upper'))
        )

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
            'load_outflow',
            'reduct_mass_lower',
            'reduct_pct_lower',
            'reduct_mass',
            'reduct_pct',
            'reduct_mass_upper',
            'reduct_pct_upper',
        ]

        final_cols = [
            'Runoff Load (lower 95%)',
            'Runoff Load',
            'Runoff Load (upper 95%',
            'Bypass Load (lower 95%)',
            'Bypass Load',
            'Bypass Load (upper 95%',
            'Estimated Total Influent Load (lower 95%)',
            'Estimated Total Influent Load',
            'Estimated Total Influent Load (upper 95%',
            'Total Effluent Load',
            'Load Reduction Mass (lower 95%)',
            'Load Reduction Percent (lower 95%)',
            'Load Reduction Mass',
            'Load Reduction Percent',
            'Load Reduction Mass (upper 95%)',
            'Load Reduction Percent (upper 95%)',

        ]

        return loads[final_cols_order].rename(columns=dict(zip(final_cols_order, final_cols)))

    def _unsampled_load_estimates(self,  sampletype='composite', excluded_dates=None, timegroup=None, NAval=None):
        """ Returns the loading estimates for unsampled storms.

        Influent concentration values from the 95% confidence intervals
        if the seasonal "influent" medians. Effluent concentrations are
        also the 95% confidence intervals around the seasonal medians
        of the actual effluent data.

        Parameters
        ----------
        sampletype : string (default = 'composite')
            The types of samples to be summarized. Valid values are
            "composite" and "grab".
        timegroup : string, optional
            Optional string that defined how results should be group
            temporally. Valid options are "season", "grouped_season",
            and year. Default behavior does no temporal grouping.
        excluded_dates : list of date-likes, optional
            This is a list of storm start dates that will be removed
            from the storms dataframe prior to computing statistics.
        NAval : float, optional
            Default value with which NA (missing) loads will be filled.
            If none, NAs will remain inplace.

        Warnings
        --------
        This should not be taken seriously. This is a total hand-wavey,
        smoke-n-mirrors hack. Do not make engineering decisions based on
        this.

        Returns
        -------
        unsampled_load_estimates : pandas.DataFrame

        """

        rename_cols = {
            'peak_precip_intensity': 'peak_precip_intensity_mm_per_hr',
            'total_precip_depth': 'total_precip_depth_mm',
            'peak_outflow': 'peak_outflow_L_per_s',
            'influent median': 'influent_median',
            'Median Effluent': 'effluent_median'
        }

        final_cols = [
            'site', 'sampletype', 'season', 'has_outflow', 'parameter', 'units',
            'influent_median', 'effluent_median', 'storm_number', 'antecedent_days',
            'start_date', 'end_date', 'duration_hours', 'peak_precip_intensity_mm_per_hr',
            'total_precip_depth_mm', 'runoff_m3', 'bypass_m3', 'inflow_m3', 'outflow_m3',
            'peak_outflow_L_per_s', 'centroid_lag_hours', 'peak_lag_hours',
            'load_units', 'load_factor', 'load_runoff', 'load_bypass',
            'load_inflow', 'load_outflow'
          ]

        index = pandas.MultiIndex.from_product(
            [self.unsampled_storms, self.tidy_data['parameter'].unique()],
            names=['storm_number', 'parameter']
        )

        def compute_load(df, volcol, conccol, conversioncol='load_factor', NAval=None):
            load = df['load_factor'] * df[volcol] * df[conccol]
            if NAval is not None:
                load = load.fillna(NAval)
            return load

        sampletyle = _check_sampletype(sampletype)
        timecol = _check_timegroup(timegroup)
        groupcols = ['site', 'sampletype', 'parameter', 'load_units', 'has_outflow']
        if timecol is not None:
            groupcols.append(timecol)

        unsampled_loads = (
            pandas.DataFrame(index=index, columns=['_junk'])
                  .reset_index()
                  .merge(self.storm_info, on='storm_number')
                  .merge(self.influentmedians, on=['parameter', 'season'])
                  .merge(self.medians('concentration', timegroup='season'), on=['parameter', 'season', 'units'])
                  .pipe(_remove_storms_from_df, excluded_dates, "start_date")
                  .assign(site=self.siteid, sampletype='unsampled')
                  .assign(load_units=lambda df: df['parameter'].apply(lambda p: info.getPOCInfo('cvcname', p, 'load_units')))
                  .assign(load_factor=lambda df: df['parameter'].apply(lambda p: info.getPOCInfo('cvcname', p, 'load_factor')))
                  .assign(load_runoff_lower=lambda df: compute_load(df, 'runoff_m3', 'influent lower'))
                  .assign(load_runoff=lambda df: compute_load(df, 'runoff_m3', 'influent median'))
                  .assign(load_runoff_upper=lambda df: compute_load(df, 'runoff_m3', 'influent upper'))
                  .assign(load_bypass_lower=lambda df: compute_load(df, 'bypass_m3', 'influent lower'))
                  .assign(load_bypass=lambda df: compute_load(df, 'bypass_m3', 'influent median'))
                  .assign(load_bypass_upper=lambda df: compute_load(df, 'bypass_m3', 'influent upper'))
                  .assign(load_inflow_lower=lambda df: compute_load(df, 'inflow_m3', 'influent lower'))
                  .assign(load_inflow=lambda df: compute_load(df, 'inflow_m3', 'influent median'))
                  .assign(load_inflow_upper=lambda df: compute_load(df, 'inflow_m3', 'influent upper'))
                  .assign(load_outflow_lower=lambda df: compute_load(df, 'outflow_m3', 'effluent lower'))
                  .assign(load_outflow=lambda df: compute_load(df, 'outflow_m3', 'effluent median'))
                  .assign(load_outflow_upper=lambda df: compute_load(df, 'outflow_m3', 'effluent upper'))
                  .select(lambda c: ('median' in c) or ('load' in c) or (c in groupcols), axis=1)
                  .groupby(by=groupcols)
                  .sum()
                  .reset_index()
                  .sort_values(by=groupcols)
        )

        return unsampled_loads

    def prevalence_table(self, sampletype='composite'):
        """ Returns a sample prevalence table for the given sample type.
        """
        sampletype = _check_sampletype(sampletype)
        pt = (
            self.wqdata
                .query("sampletype == @sampletype")
                .groupby(by=['season', 'samplestart', 'parameter', 'sampletype'])
                .count()['concentration']
                .unstack(level='sampletype')
                .unstack(level='parameter')
                .reset_index()
        )
        return pt

    def hydro_jointplot(self, xcol, ycol, conditions=None, one2one=True):
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

        if conditions is None:
            data = self.storm_info
        else:
            data = self.storm_info.query(conditions)

        jg = utils.figutils.jointplot(
            x=xcol, y=ycol, data=data, one2one=one2one, color=self.color,
            xlabel=column_labels[xcol], ylabel=column_labels[ycol],
        )

        figname = '{}-HydroJoinPlot_{}_vs_{}'.format(self.siteid, xcol, ycol)

        viz._savefig(jg.fig, figname, extra='HydroJointPlot')

    def hydro_pairplot(self, by='season', palette=None):
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
            self.storm_info
                .assign(year=self.storm_info['start_date'].dt.year)
                .rename(columns={'has_outflow': 'Has outflow?', 'grouped_season': 'Seasons'})
                .select(lambda c: c in cols, axis=1)
                .rename(columns=lambda c: c.replace('_', ' '))
        )

        if by == 'season':
            pg = seaborn.pairplot(
                sinfo,
                palette=palette or 'BrBG_r',
                hue='season',
                markers=['o', 's', '^', 'd'],
                hue_order=['winter', 'spring', 'summer', 'autumn'],
                vars=var_cols
            )

        elif by =='year':
            pg = seaborn.pairplot(
                sinfo,
                hue='year',
                palette=palette or 'deep',
                vars=var_cols
            )

        elif by == 'outflow':
            pg = seaborn.pairplot(
                sinfo,
                hue='Has outflow?',
                palette=palette or 'deep',
                markers=['o', 's'],
                hue_order=['Yes', 'No'],
                vars=var_cols
            )

        elif by == 'grouped_season':
            pg = seaborn.pairplot(
                sinfo,
                palette=palette or 'BrBG_r',
                hue='Seasons',
                markers=['o', '^'],
                hue_order=['winter/spring', 'summer/autumn'],
                vars=var_cols
            )

        for ax in pg.axes.flat:
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        figname = '{}-HydroPairPlot_by_{}'.format(self.siteid, by)
        viz._savefig(pg.fig, figname, extra='HydroPairPlot')

    def allISRs(self, sampletype, version='draft'):
        """ Compiles all Individual Storm Reports for a site

        Parameters
        ---------
        sampletype : string
            Type of samples whose reports should be generated.
            Valid values are "grab" or "composite". Only "composite" is
            recommended.
        version : string, optional (default = 'draft')
            Whether the file should be marked as "draft" or "final"

        Returns
        -------
        None

        """

        if not os.path.exists(self.templateISR):
            raise IOError("template at {} not found".format(self.templateISR))

        # XXX: hack to load up storm info
        _ = self.storms

        for sample in self.samples[sampletype]:
            if sample.storm is not None:
                sample.templateISR = self.templateISR
                tex = sample.compileISR(version=version, clean=True)


@np.deprecate
def normalize_units(dataframe, units_map, targetunit, paramcol='Parameter',
                    rescol='Outflow_res', unitcol='Outflow_unit', debug=False):

    # standardize units in the wqdata
    dataframe['normalize'] = dataframe[unitcol].map(units_map.get)
    if isinstance(targetunit, dict):
        dataframe['targetunit'] = dataframe[paramcol].map(targetunit.get)
    else:
        dataframe['targetunit'] = targetunit

    dataframe['convert'] = dataframe['targetunit'].map(units_map.get)
    dataframe[rescol] = dataframe[rescol] * dataframe['normalize'] / dataframe['convert']

    # reassign unites
    dataframe[unitcol] = dataframe.targetunit
    return dataframe
