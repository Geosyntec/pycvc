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


class _external_source(object):
    def boxplot(self, ax, position, xlabels, **selection_kwds):
        for label in self.labels:
            position += 1
            xlabels.append(label.replace('/', r'/\\'))
            selection_kwds[self.label_col] = label
            effluent = self.datacollection.selectLocations(**selection_kwds)

            # if there's enough BMP data, do the boxplots
            if effluent is not None and effluent.hasData:
                effluent.name = label
                effluent.color = self.color
                effluent.plot_marker = self.marker
                effluent.boxplot(ax=ax, pos=position, patch_artist=True)

        return position, xlabels


class nsqd(_external_source):
    def __init__(self, color, marker):
        self.color = color
        self.marker = marker
        self._data = None
        self._medians = None
        self._datacollection = None
        self.label_col = 'primary_landuse'

    @property
    def landuses(self):
        return self.data['primary_landuse'].unique()

    @property
    def labels(self):
        return self.landuses

    @property
    def data(self):
        params = [p['nsqdname'] for p in POC_dicts]
        if self._data is None:
            self._data = (
                pynsqd.NSQData()
                      .data
                      .query("parameter in @params")
                      .query("fraction == 'Total'")
                      .query("primary_landuse != 'Unknown'")
                      .query("epa_rain_zone == 1")
                      .assign(station='outflow')
                      .assign(cvcparam=lambda df: df['parameter'].apply(self._get_cvc_parameter))
                      .assign(season=lambda df: df['start_date'].apply(wqio.utils.getSeason))
            )
        return self._data

    @property
    def datacollection(self):
        if self._datacollection is None:
            indexcols = [
                'epa_rain_zone',
                'location_code',
                'station_name',
                'primary_landuse',
                'start_date',
                'season',
                'station',
                'cvcparam',
                'units',
            ]
            groupcols = ['primary_landuse', 'season']
            d = self.data.groupby(by=indexcols).first()
            dc = wqio.DataCollection(d, ndval='<', othergroups=groupcols, paramcol='cvcparam')

            self._datacollection = dc
        return  self._datacollection

    @property
    def medians(self):
        if self._medians is None:
            self._medians = (
                self.datacollection
                   .medians['outflow']
                   .xs(['Residential'], level=['primary_landuse'])
                   .pipe(np.round, 3)
                   .reset_index()
                   .rename(columns={'cvcparam': 'parameter', 'stat': 'NSQD Medians'})
                )

        return self._medians

    @staticmethod
    def _get_cvc_parameter(nsqdparam):
        try:
            cvcparam = list(filter(
                lambda p: p['nsqdname'] == nsqdparam, POC_dicts
            ))[0]['cvcname']
        except IndexError:
            cvcparam = np.nan
        return cvcparam


class bmpdb(_external_source):
    def __init__(self, color, marker):
        self.color = color
        self.marker = marker
        self.paramnames = [p['bmpname'] for p in POC_dicts]
        self._mainparams = list(filter(
            lambda x: x['conc_units']['plain'] != 'CFU/100 mL', POC_dicts
        ))
        self._bioparams = list(filter(
            lambda x: x['conc_units']['plain'] == 'CFU/100 mL', POC_dicts
        ))

        self.table, self.db = pybmpdb.getSummaryData(
            catanalysis=False, astable=True,
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
            self._data = self.datacollection.tidy.copy()
        return self._data

    @property
    def datacollection(self):
        if self._datacollection is None:
            groups = ['category', 'units']
            self._datacollection = self.table.to_DataCollection(othergroups=groups)
        return self._datacollection

    @property
    def medians(self):
        if self._medians is None:

            medians = (
                self.data
                    .groupby(by=['station', 'parameter', 'units', 'category'])['res']
                    .median()
                    .xs('outflow', level='station')
                    .unstack(level='category')
                    .reset_index()
                    .rename(columns={'parameter': '_p'})
            )
            medians['parameter'] = medians['_p'].apply(self._get_cvc_parameter)

            self._medians = (
                medians.dropna(subset=['parameter'])
                       .drop('_p', axis=1)
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


def loadExternalData(colors, markers):
    return _nsqd(colors[0], markers[0]), _bmpdb(colors[1], markers[1])
