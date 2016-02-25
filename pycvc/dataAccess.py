import os
import sys
import glob
from pkg_resources import resource_filename

import numpy as np
from scipy import interpolate
import pandas
import pyodbc

import wqio
from wqio import utils

# CVC project info
from . import info
from . import viz
from . import validate

# CVC-specific wqio.events subclasses
from .samples import GrabSample, CompositeSample, Storm


__all__ = ['Database', 'Site']


LITERS_PER_CUBICMETER = 1000.
MILLIMETERS_PER_METER = 1000.


def _fix_cvc_bacteria_units(df, unitscol='units'):
    df = df.copy()
    df[unitscol] = df[unitscol].replace(to_replace='CFU/100mL', value='CFU/100 mL')
    return df


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
    """

    def __init__(self, dbfile, nsqdata=None, bmpdb=None):
        self.dbfile = dbfile
        self.nsqdata = nsqdata
        self.bmpdb = bmpdb
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
            self._sites = self._run_query("select * from sites")
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
        self._tidy_wq = None
        self._tidy_hydro = None
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
            self._wqdata = (
                self.db
                    .getWQData(self.siteid, onlyPOCs=self.onlyPOCs)
                    .assign(season=lambda df: df['samplestart'].apply(utils.getSeason))
                    .assign(grouped_season=lambda df: df['samplestart'].apply(_grouped_seasons))
                    .assign(year=lambda df: df['samplestart'].dt.year.astype(str))
                    .assign(sampletype=lambda df: df['sampletype'].str.lower())
                    .pipe(_fix_cvc_bacteria_units)
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
    def tidy_wq(self):

        def compute_load(row, volcol, conccol, backupcol=None, conversioncol='load_factor', NAval=None):
            if pandas.isnull(row[conccol]) and backupcol is not None:
                conccol = backupcol

            load = row[conversioncol] * row[volcol] * row[conccol]
            if NAval is not None and np.isnan(load):
                load = NAval

            return load

        if self._tidy_wq is None:
            wq = (
                self.wqdata
                   .query("sampletype == 'composite'")
                   .assign(storm_number=lambda df: df['samplestart'].apply(lambda d: self._get_storm_from_date(d)[0]))
            )

            index = pandas.MultiIndex.from_product(
                [self.storm_info['storm_number'], self.wqdata['parameter'].unique()],
                names=['storm_number', 'parameter']
            )

            final_cols = [
                'site', 'year', 'season', 'grouped_season', 'storm_number',
                'antecedent_days', 'start_date', 'end_date', 'duration_hours',
                'peak_precip_intensity', 'total_precip_depth', 'runoff_m3',
                'bypass_m3', 'inflow_m3', 'outflow_m3', 'outflow_mm',
                'peak_outflow', 'centroid_lag_hours', 'peak_lag_hours',
                'has_outflow', 'sm_est_peak_inflow', 'parameter', 'units',
                'sample', 'sampletype', 'samplestart', 'samplestop', 'interval_minutes',
                'detectionlimit', 'qualifier', 'concentration',
                'influent lower', 'influent median', 'influent upper',
                'effluent lower', 'effluent median', 'effluent upper',
                'load_units', 'load_factor',
                'load_runoff_lower', 'load_runoff', 'load_runoff_upper',
                'load_bypass_lower', 'load_bypass', 'load_bypass_upper',
                'load_inflow_lower', 'load_inflow', 'load_inflow_upper',
                'load_outflow_lower', 'load_outflow', 'load_outflow_upper',
            ]

            self._tidy_wq = (
                pandas.DataFrame(index=index, columns=['_junk'])
                      .reset_index()
                      .drop('_junk', axis=1)
                      .assign(units=lambda df: df['parameter'].apply(lambda x: info.getPOCInfo('cvcname', x, 'conc_units')['plain']))
                      .merge(self.storm_info, on='storm_number', how='outer')
                      .merge(self.influentmedians, on=['parameter', 'season', 'units'], how='outer')
                      .merge(self.effluent('concentration', groupby_col='season'), on=['parameter', 'season', 'units'], how='outer')
                      .merge(wq, on=['parameter', 'storm_number', 'site', 'year', 'season', 'grouped_season', 'units'], how='outer')
                      .assign(load_units=lambda df: df['parameter'].apply(lambda p: info.getPOCInfo('cvcname', p, 'load_units')))
                      .assign(load_factor=lambda df: df['parameter'].apply(lambda p: info.getPOCInfo('cvcname', p, 'load_factor')))
                      .assign(load_runoff_lower=lambda df: df.apply(lambda row: compute_load(row, 'runoff_m3', 'influent lower'), axis=1))
                      .assign(load_runoff=lambda df: df.apply(lambda row: compute_load(row, 'runoff_m3', 'influent median'), axis=1))
                      .assign(load_runoff_upper=lambda df: df.apply(lambda row: compute_load(row, 'runoff_m3', 'influent upper'), axis=1))
                      .assign(load_bypass_lower=lambda df: df.apply(lambda row: compute_load(row, 'bypass_m3', 'influent lower'), axis=1))
                      .assign(load_bypass=lambda df: df.apply(lambda row: compute_load(row, 'bypass_m3', 'influent median'), axis=1))
                      .assign(load_bypass_upper=lambda df: df.apply(lambda row: compute_load(row, 'bypass_m3', 'influent upper'), axis=1))
                      .assign(load_inflow_lower=lambda df: df.apply(lambda row: compute_load(row, 'inflow_m3', 'influent lower'), axis=1))
                      .assign(load_inflow=lambda df: df.apply(lambda row: compute_load(row, 'inflow_m3', 'influent median'), axis=1))
                      .assign(load_inflow_upper=lambda df: df.apply(lambda row: compute_load(row, 'inflow_m3', 'influent upper'), axis=1))
                      .assign(load_outflow_lower=lambda df: df.apply(lambda row: compute_load(row, 'outflow_m3', 'concentration', backupcol='effluent lower'), axis=1))
                      .assign(load_outflow=lambda df: df.apply(lambda row: compute_load(row, 'outflow_m3', 'concentration', backupcol='effluent median'), axis=1))
                      .assign(load_outflow_upper=lambda df: df.apply(lambda row: compute_load(row, 'outflow_m3', 'concentration', backupcol='effluent upper'), axis=1))
                      .fillna(value={'sampletype': 'unsampled'})
            )

        return self._tidy_wq

    @property
    def tidy_hydro(self):
        return self.storm_info

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
        sampletype = validate.sampletype(sampletype)

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
        sampletype = validate.sampletype(sampletype)
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

    def _make_dc(self, rescol, sampletype, groupby_col=None):
        sampletype = validate.sampletype(sampletype)
        rescol, unitscol = validate.rescol(rescol)
        timecol = validate.groupby_col(groupby_col)

        if timecol is None:
            othergroups = [unitscol]
        else:
            othergroups = [unitscol, timecol]

        dc_options = dict(rescol=rescol, qualcol='qualifier', ndval='<',
                          othergroups=othergroups, stationcol='site')
        dc = (
            self.wqdata
                .query("sampletype == @sampletype")
                .reset_index()
                .pipe(wqio.DataCollection, **dc_options)
        )

        return dc

    def medians(self, rescol='concentration', sampletype='composite', groupby_col=None):
        """ Returns a DataFrame of the WQ medians for the given
        sampletype.

        Parameters
        ----------
        rescol : string (default = 'concentration')
            The result column to summarize. Valid values are
            "concentration" and "load_outflow".
        sampletype : string (default = 'composite')
            The types of samples to be summarized. Valid values are
            "composite" and "grab".
        groupby_col : string, optional
            Optional string that defined how results should be group
            temporally. Valid options are "season", "grouped_season",
            and "year". Default behavior does no temporal grouping.

        Returns
        -------
        medians : pandas.DataFrame

        """

        medians = (
            self._make_dc(rescol, sampletype, groupby_col=groupby_col)
                .medians[self.siteid]
                .rename(columns={'stat': 'median'})
                .rename(columns=lambda c: 'effluent {}'.format(c))
                .reset_index()
        )
        return medians

    def effluent(self, rescol='concentration', sampletype='composite', groupby_col=None):
        """ Returns a DataFrame of the characteristic effluent WQ for
        the given sampletype.

        Parameters
        ----------
        rescol : string (default = 'concentration')
            The result column to summarize. Valid values are
            "concentration" and "load_outflow".
        sampletype : string (default = 'composite')
            The types of samples to be summarized. Valid values are
            "composite" and "grab".
        groupby_col : string, optional
            Optional string that defined how results should be group
            temporally. Valid options are "season", "grouped_season",
            and "year". Default behavior does no temporal grouping.

        Returns
        -------
        effluent : pandas.DataFrame

        """
        def notches(med, IQR, N):
            notch_min = med - 1.57 * IQR / np.sqrt(N)
            notch_max = med + 1.57 * IQR / np.sqrt(N)
            final_cols = ['effluent lower', 'effluent median', 'effluent upper']
            notches = (
                notch_min.join(notch_max, lsuffix='_lower', rsuffix='_upper')
                         .join(med.rename(columns={'ros_concentration': 'effluent median'}))
                         .rename(columns=lambda c: c.replace('ros_concentration_', 'effluent '))
                         [final_cols]
                         .reset_index()
            )
            return notches

        if self.compdates.shape[0] > 10:
            return self.medians(rescol=rescol, sampletype=sampletype, groupby_col=groupby_col)
        else:
            dc = self._make_dc(rescol, sampletype, groupby_col=groupby_col)

            # use the 5/95 percentiles
            statcol = 'ros_concentration'
            lower = dc.percentiles(5)[self.siteid].rename(columns={statcol: 'effluent lower'})
            med = dc.percentiles(50)[self.siteid].rename(columns={statcol: 'effluent median'})
            upper = dc.percentiles(95)[self.siteid].rename(columns={statcol: 'effluent upper'})
            return lower.join(med).join(upper).reset_index()

            ## uses the normal approximation
            # med = dc.percentiles(50)[self.siteid]
            # IQR = (dc.percentiles(75) - dc.percentiles(25))[self.siteid]
            # counts = dc.count[self.siteid]
            # return notches(med, IQR, counts)

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
