import sys
import os

import numpy as np
import pandas

import wqio
from wqio import utils
import pybmpdb
import pynsqd

from .info import POC_dicts


bmpcats_to_use = [
    'Bioretention', 'Detention Basin',
    'Green Roof', 'Biofilter',
    'LID', 'Manufactured Device',
    'Media Filter', 'Porous Pavement',
    'Retention Pond', 'Wetland Basin',
    'Wetland Channel'
]


def _fix_nsqd_bacteria_units(df, unitscol='units'):
    df = df.copy()
    df[unitscol] = df[unitscol].replace(to_replace='MPN/100 mL', value='CFU/100 mL')
    return df


class nsqd:
    def __init__(self, color, marker, datafile=None):
        self.color = color
        self.marker = marker
        self._data = None
        self._datacollection = None
        self._medians = None
        self._seasonal_datacollection = None
        self._seasonal_medians = None
        self.label_col = 'primary_landuse'
        self.index_cols = [
            'epa_rain_zone', 'location_code', 'station_name', 'primary_landuse',
            'start_date', 'season', 'station', 'parameter', 'units',
        ]
        self.datafile = datafile
        self.db = pynsqd.NSQData(datapath=self.datafile)

    @property
    def landuses(self):
        return self.data['primary_landuse'].unique()

    @property
    def labels(self):
        return self.landuses

    @property
    def data(self):
        if self._data is None:
            params = [p['nsqdname'] for p in POC_dicts]
            self._data = (
                self.db
                    .data
                    .query("primary_landuse != 'Unknown'")
                    .query("parameter in @params")
                    .query("fraction == 'Total'")
                    .query("epa_rain_zone == 1")
                    .assign(cvcparam=lambda df: df['parameter'].apply(self._get_cvc_parameter))
                    .drop('parameter', axis=1)
                    .rename(columns={'cvcparam': 'parameter'})
                    .groupby(by=self.index_cols)
                    .first()
                    .reset_index()
                    .pipe(_fix_nsqd_bacteria_units)
            )
        return self._data


    def _make_dc(self, which):
        _dc_map = {
            'overall': ['units', 'primary_landuse'],
            'seasonal': ['units', 'primary_landuse', 'season'],
        }

        dc = wqio.DataCollection(
            self.data.set_index(self.index_cols),
            ndval='<',
            othergroups=_dc_map[which],
            paramcol='parameter'
        )

        return dc


    def _get_medians(self, which):
        _med_dict = {
            'overall': self.datacollection.medians,
            'seasonal': self.seasonal_datacollection.medians,
        }

        medians = (
            _med_dict[which.lower()]
                ['outflow']
                .xs('Residential', level='primary_landuse')
                .pipe(np.round, 3)
                .reset_index()
                .rename(columns={'stat': 'NSQD Medians'})
        )

        return medians

    @property
    def datacollection(self):
        if self._datacollection is None:
            self._datacollection = self._make_dc('overall')
        return  self._datacollection

    @property
    def medians(self):
        if self._medians is None:
            self._medians = self._get_medians('overall')

        return self._medians

    @property
    def seasonal_datacollection(self):
        if self._seasonal_datacollection is None:
            self._seasonal_datacollection = self._make_dc('seasonal')
        return  self._seasonal_datacollection

    @property
    def seasonal_medians(self):
        if self._seasonal_medians is None:
            self._seasonal_medians = self._get_medians('seasonal')

        return self._seasonal_medians

    @staticmethod
    def _get_cvc_parameter(nsqdparam):
        try:
            cvcparam = list(filter(
                lambda p: p['nsqdname'] == nsqdparam, POC_dicts
            ))[0]['cvcname']
        except IndexError:
            cvcparam = np.nan
        return cvcparam


class bmpdb:
    def __init__(self, color, marker, datafile=None):
        self.color = color
        self.marker = marker
        self.paramnames = [p['bmpname'] for p in POC_dicts]
        self._mainparams = list(filter(
            lambda x: x['conc_units']['plain'] != 'CFU/100 mL', POC_dicts
        ))
        self._bioparams = list(filter(
            lambda x: x['conc_units']['plain'] == 'CFU/100 mL', POC_dicts
        ))

        self.datafile = datafile
        self.table, self.db = pybmpdb.getSummaryData(
            dbpath=self.datafile,
            catanalysis=False,
            astable=True,
            parameter=self.paramnames,
            category=bmpcats_to_use,
            epazone=1,
        )

        self._data = None
        self._datasets = None
        self._effluentLocations = None
        self._medians = None
        self._datacollection = None
        self.label_col = 'category'

    @property
    def categories(self):
        return self.table.bmp_categories

    @property
    def labels(self):
        return self.categories

    @property
    def data(self):
        if self._data is None:
            index_cache = self.table.data.index.names
            self._data = (
                self.table
                    .data
                    .reset_index()
                    .query("station == 'outflow'")
                    .query("epazone == 1")
                    .assign(bmpparam=lambda df: df['parameter'].apply(self._get_cvc_parameter))
                    .drop('parameter', axis=1)
                    .rename(columns={'bmpparam': 'parameter'})
                    .set_index(index_cache)
            )
        return self._data

    @property
    def datacollection(self):
        if self._datacollection is None:
            groupcols = ['units', 'category']
            dc = wqio.DataCollection(self.data, ndval='ND', othergroups=groupcols,
                                     paramcol='parameter')

            self._datacollection = dc
        return  self._datacollection

    @property
    def medians(self):
        if self._medians is None:
            self._medians = (
                self.datacollection
                    .medians['outflow']
                    .pipe(np.round, 3)
                    .reset_index()
                    .rename(columns={'stat': 'BMPDB Medians'})
            )
        return self._medians


    @staticmethod
    def _get_cvc_parameter(bmpparam):
        try:
            bmpparam = list(filter(
                lambda p: p['bmpname'] == bmpparam, POC_dicts
            ))[0]['cvcname']
        except:
            bmpparam = np.nan

        return bmpparam


def combine_wq(wq, external, external_site_col):
    final_cols = [
        'parameter',
        'units',
        'site',
        'concentration',
    ]
    exttidy = (
        external.datacollection.tidy
            .rename(columns={external_site_col: 'site', 'ros_res': 'concentration'})
    )[final_cols]

    tidy = pandas.concat([wq[final_cols], exttidy])
    return tidy
