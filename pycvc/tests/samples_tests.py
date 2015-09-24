import os
import sys
import datetime
from pkg_resources import resource_filename

import numpy as np
import matplotlib.pyplot as plt
import pandas

import nose.tools as nt
import numpy.testing as nptest
from matplotlib.testing.decorators import image_comparison, cleanup
import pandas.util.testing as pdtest

import wqio
from pycvc import samples, info

from wqio.tests.core_tests import samples_tests, hydro_tests
from wqio.tests.utils_tests import misc_tests


class test_LaTeXDirecory(misc_tests.test_LaTeXDirecory):
    pass


class _wq_sample_mixin(object):
    def test__res_with_units(self):
        nt.assert_equal(self.wqs._res_with_units(0.12, 'mg/L'), '0.120 mg/L')
        nt.assert_equal(self.wqs._res_with_units(np.nan, 'mg/L'), '--')

    def test_siteid(self):
        nt.assert_true(hasattr(self.wqs, 'siteid'))
        nt.assert_equal(self.wqs.siteid, self.known_siteid)

    def test_general_tex_table(self):
        nt.assert_true(hasattr(self.wqs, 'general_tex_table'))
        nt.assert_equal(self.wqs.general_tex_table, self.known_general_tex_table)

    def test_hydro_tex_table(self):
        nt.assert_true(hasattr(self.wqs, 'hydro_tex_table'))
        nt.assert_equal(self.wqs.hydro_tex_table, self.known_hydro_tex_table)

    def test_wq_tex_table(self):
        nt.assert_true(hasattr(self.wqs, 'wq_tex_table'))
        nt.assert_equal(self.wqs.wq_tex_table, self.known_wq_tex_table)

    def test_storm_figure(self):
        nt.assert_true(hasattr(self.wqs, 'storm_figure'))
        nt.assert_equal(self.wqs.storm_figure, self.known_storm_figure)

    def test_other_props(self):
        nt.assert_true(hasattr(self.wqs, 'templateISR'))
        self.wqs.templateISR = 'test'
        nt.assert_equal(self.wqs.templateISR, 'test')

        nt.assert_true(hasattr(self.wqs, 'tocentry'))
        self.wqs.tocentry = 'test'
        nt.assert_equal(self.wqs.tocentry, 'test')

        nt.assert_true(hasattr(self.wqs, 'wqstd'))
        self.wqs.wqstd = 'test'
        nt.assert_equal(self.wqs.wqstd, 'test')

    def test_wq_table(self):
        df = self.wqs.wq_table(writeToFiles=False)
        pdtest.assert_frame_equal(df, self.known_wqtable[df.columns])


class test_CompositeSample(samples_tests.test_CompositeSample_NoStorm, _wq_sample_mixin):
    def setup(self):
        self.basic_setup()
        self.known_siteid = 'Test Site'
        self.known_general_tex_table = 'TestSite-2013-02-24-1659-1-General'
        self.known_hydro_tex_table = 'TestSite-2013-02-24-1659-2-Hydro'
        self.known_wq_tex_table = 'TestSite-2013-02-24-1659-3-WQComposite'
        self.known_storm_figure = 'TestSite-2013-02-24-1659-Composite'
        self.test_name = 'CompositeNoStorm'
        self.known_yfactor = 0.25
        self.known_starttime = '2013-02-24 16:59'
        self.known_endtime = '2013-02-25 02:59'
        self.known_season = 'winter'
        self.known_sample_ts_len = 31
        self.known_samplefreq = pandas.tseries.offsets.Minute(20)
        self.known_samplefreq_type = pandas.tseries.offsets.Minute
        self.known_marker = 'x'
        self.known_label = 'Composite Sample'
        self.known_wqtable = pandas.DataFrame({
            'Effluent EMC': {
                0: '786 ug/L', 1: '0.160 ug/L', 2: '8.60 ug/L', 3: '140 mg/L',
                4: '1,350 ug/L', 5: '6.13 ug/L', 6: '2.20 ug/L', 7: '1.40 mg/L',
                8: '1.50 mg/L', 9: '0.0520 mg/L', 10: '1.20 mg/L', 11: '0.500 mg/L',
                12: '0.130 mg/L', 13: '110 mg/L', 14: '43.4 ug/L'
            },
            'Detection Limit': {
                0: '0.500 ug/L', 1: '0.0100 ug/L', 2: '0.200 ug/L', 3: '1.00 mg/L',
                4: '5.00 ug/L', 5: '0.0500 ug/L', 6: '0.200 ug/L', 7: '0.100 mg/L',
                8: '0.100 mg/L', 9: '0.00200 mg/L', 10: '0.100 mg/L', 11: '0.500 mg/L',
                12: '0.100 mg/L', 13: '3.00 mg/L', 14: '0.500 ug/L'
            },
            'Effluent Load': {
                0: '40.6 g', 1: '0.00827 g', 2: '0.445 g', 3: '7,240 g',
                4: '69.8 g', 5: '0.317 g', 6: '0.114 g', 7: '72.4 g',
                8: '77.6 g', 9: '2.69 g', 10: '62.0 g', 11: '0.0259 g',
                12: '6.72 g', 13: '5,690 g', 14: '2.24 g'},
            'WQ Guideline': {
                0: '10.0 ug/L', 1: '10.0 ug/L', 2: '10.0 ug/L', 3: '10.0 mg/L',
                4: '10.0 ug/L', 5: '10.0 ug/L', 6: '10.0 ug/L', 7: '10.0 mg/L',
                8: '10.0 mg/L', 9: '10.0 mg/L', 10: '10.0 mg/L', 11: '10.0 mg/L',
                12: '10.0 mg/L', 13: '10.0 mg/L', 14: '10.0 ug/L'
            },
            'Parameter': {
                0: 'Aluminum (Al)', 1: 'Cadmium (Cd)', 2: 'Copper (Cu)', 3: 'Dissolved Chloride (Cl)',
                4: 'Iron (Fe)', 5: 'Lead (Pb)', 6: 'Nickel (Ni)', 7: 'Nitrate (N)',
                8: 'Nitrate + Nitrite', 9: 'Orthophosphate (P)', 10: 'Total Kjeldahl Nitrogen (TKN)', 11: 'Total Oil & Grease',
                12: 'Total Phosphorus', 13: 'Total Suspended Solids', 14: 'Zinc (Zn)'
            }
        })
        data = pandas.DataFrame({
            'concentration': {
                0: 786.0, 1: 0.16, 2: 8.60, 3: 140.0,
                4: 9000.0, 5: 1350.0, 6: 6.13, 7: 2.20,
                8: 1.40, 9: 1.5, 10: 0.052, 11: 1.2,
                12: 0.5, 13: 0.13, 14: 110.0, 15: 43.4
            },
            'detectionlimit': {
                0: 0.5, 1: 0.01, 2: 0.20, 3: 1.0,
                4: np.nan, 5: 5.0, 6: 0.05, 7: 0.2,
                8: 0.1, 9: 0.1, 10: 0.002, 11: 0.1,
                12: 0.5, 13: 0.1, 14: 3.0, 15: 0.5
            },
            'load_outflow': {
                0: 40.642016399999555, 1: 0.0082731839999999109,
                2: 0.44468363999999516, 3: 7239.0359999999218,
                4: 4653665999.9999495, 5: 69.80498999999925,
                6: 0.31696636199999656, 7: 0.11375627999999877,
                8: 72.390359999999205, 9: 77.561099999999158,
                10: 2.6887847999999708, 11: 62.048879999999322,
                12: 0.025853699999999719, 13: 6.7219619999999276,
                14: 5687.8139999999385, 15: 2.2441011599999756
            },
            'load_units': {
                0: 'g', 1: 'g', 2: 'g', 3: 'g',
                4: 'CFU', 5: 'g', 6: 'g', 7: 'g',
                8: 'g', 9: 'g', 10: 'g', 11: 'g',
                12: 'g', 13: 'g', 14: 'g', 15: 'g'
            },
            'parameter': {
                0: 'Aluminum (Al)', 1: 'Cadmium (Cd)',
                2: 'Copper (Cu)', 3: 'Dissolved Chloride (Cl)',
                4: 'Escherichia coli', 5: 'Iron (Fe)',
                6: 'Lead (Pb)', 7: 'Nickel (Ni)',
                8: 'Nitrate (N)', 9: 'Nitrate + Nitrite',
                10: 'Orthophosphate (P)', 11: 'Total Kjeldahl Nitrogen (TKN)',
                12: 'Total Oil & Grease', 13: 'Total Phosphorus',
                14: 'Total Suspended Solids', 15: 'Zinc (Zn)'
            },
            'units': {
                0: 'ug/L', 1: 'ug/L', 2: 'ug/L', 3: 'mg/L',
                4: 'CFU/100mL', 5: 'ug/L', 6: 'ug/L', 7: 'ug/L',
                8: 'mg/L', 9: 'mg/L', 10: 'mg/L', 11: 'mg/L',
                12: 'mg/L', 13: 'mg/L', 14: 'mg/L', 15: 'ug/L'
            }
        })
        self.wqs = samples.CompositeSample(data, self.known_starttime,
                                           endtime=self.known_endtime,
                                           samplefreq=self.known_samplefreq,
                                           storm=None)
        self.wqs.siteid = self.known_siteid
        self.wqs.label = self.known_label
        self.wqs.wqstd = (
            info.wqstd_template()
                .assign(upper_limit=10)
                .query("season == 'summer'")
        )


class test_GrabSample(samples_tests.test_GrabSample_NoStorm, _wq_sample_mixin):
    def setup(self):
        self.basic_setup()
        self.known_siteid = 'Test Site'
        self.known_general_tex_table = 'TestSite-2013-02-24-1659-1-General'
        self.known_hydro_tex_table = 'TestSite-2013-02-24-1659-2-Hydro'
        self.known_wq_tex_table = 'TestSite-2013-02-24-1659-3-WQGrab'
        self.known_storm_figure = 'TestSite-2013-02-24-1659-Grab'
        self.test_name = 'GrabNoStorm'
        self.known_yfactor = 0.25
        self.known_starttime = '2013-02-24 16:59'
        self.known_endtime = '2013-02-25 02:59'
        self.known_season = 'winter'
        self.known_sample_ts_len = 2
        self.known_samplefreq = None
        self.known_samplefreq_type = type(None)
        self.known_marker = '+'
        self.known_label = 'Grab Sample'
        self.known_wqtable = pandas.DataFrame({
            'Effluent EMC': {
                0: '786 ug/L', 1: '0.160 ug/L', 2: '8.60 ug/L', 3: '140 mg/L',
                4: '1,350 ug/L', 5: '6.13 ug/L', 6: '2.20 ug/L', 7: '1.40 mg/L',
                8: '1.50 mg/L', 9: '0.0520 mg/L', 10: '1.20 mg/L', 11: '0.500 mg/L',
                12: '0.130 mg/L', 13: '110 mg/L', 14: '43.4 ug/L'
            },
            'Detection Limit': {
                0: '0.500 ug/L', 1: '0.0100 ug/L', 2: '0.200 ug/L', 3: '1.00 mg/L',
                4: '5.00 ug/L', 5: '0.0500 ug/L', 6: '0.200 ug/L', 7: '0.100 mg/L',
                8: '0.100 mg/L', 9: '0.00200 mg/L', 10: '0.100 mg/L', 11: '0.500 mg/L',
                12: '0.100 mg/L', 13: '3.00 mg/L', 14: '0.500 ug/L'
            },
            'Effluent Load': {
                0: '40.6 g', 1: '0.00827 g', 2: '0.445 g', 3: '7,240 g',
                4: '69.8 g', 5: '0.317 g', 6: '0.114 g', 7: '72.4 g',
                8: '77.6 g', 9: '2.69 g', 10: '62.0 g', 11: '0.0259 g',
                12: '6.72 g', 13: '5,690 g', 14: '2.24 g'},
            'WQ Guideline': {
                0: '10.0 ug/L', 1: '10.0 ug/L', 2: '10.0 ug/L', 3: '10.0 mg/L',
                4: '10.0 ug/L', 5: '10.0 ug/L', 6: '10.0 ug/L', 7: '10.0 mg/L',
                8: '10.0 mg/L', 9: '10.0 mg/L', 10: '10.0 mg/L', 11: '10.0 mg/L',
                12: '10.0 mg/L', 13: '10.0 mg/L', 14: '10.0 ug/L'
            },
            'Parameter': {
                0: 'Aluminum (Al)', 1: 'Cadmium (Cd)', 2: 'Copper (Cu)', 3: 'Dissolved Chloride (Cl)',
                4: 'Iron (Fe)', 5: 'Lead (Pb)', 6: 'Nickel (Ni)', 7: 'Nitrate (N)',
                8: 'Nitrate + Nitrite', 9: 'Orthophosphate (P)', 10: 'Total Kjeldahl Nitrogen (TKN)', 11: 'Total Oil & Grease',
                12: 'Total Phosphorus', 13: 'Total Suspended Solids', 14: 'Zinc (Zn)'
            }
        })
        data = pandas.DataFrame({
            'concentration': {
                0: 786.0, 1: 0.16, 2: 8.60, 3: 140.0,
                4: 9000.0, 5: 1350.0, 6: 6.13, 7: 2.20,
                8: 1.40, 9: 1.5, 10: 0.052, 11: 1.2,
                12: 0.5, 13: 0.13, 14: 110.0, 15: 43.4
            },
            'detectionlimit': {
                0: 0.5, 1: 0.01, 2: 0.20, 3: 1.0,
                4: np.nan, 5: 5.0, 6: 0.05, 7: 0.2,
                8: 0.1, 9: 0.1, 10: 0.002, 11: 0.1,
                12: 0.5, 13: 0.1, 14: 3.0, 15: 0.5
            },
            'load_outflow': {
                0: 40.642016399999555, 1: 0.0082731839999999109,
                2: 0.44468363999999516, 3: 7239.0359999999218,
                4: 4653665999.9999495, 5: 69.80498999999925,
                6: 0.31696636199999656, 7: 0.11375627999999877,
                8: 72.390359999999205, 9: 77.561099999999158,
                10: 2.6887847999999708, 11: 62.048879999999322,
                12: 0.025853699999999719, 13: 6.7219619999999276,
                14: 5687.8139999999385, 15: 2.2441011599999756
            },
            'load_units': {
                0: 'g', 1: 'g', 2: 'g', 3: 'g',
                4: 'CFU', 5: 'g', 6: 'g', 7: 'g',
                8: 'g', 9: 'g', 10: 'g', 11: 'g',
                12: 'g', 13: 'g', 14: 'g', 15: 'g'
            },
            'parameter': {
                0: 'Aluminum (Al)', 1: 'Cadmium (Cd)',
                2: 'Copper (Cu)', 3: 'Dissolved Chloride (Cl)',
                4: 'Escherichia coli', 5: 'Iron (Fe)',
                6: 'Lead (Pb)', 7: 'Nickel (Ni)',
                8: 'Nitrate (N)', 9: 'Nitrate + Nitrite',
                10: 'Orthophosphate (P)', 11: 'Total Kjeldahl Nitrogen (TKN)',
                12: 'Total Oil & Grease', 13: 'Total Phosphorus',
                14: 'Total Suspended Solids', 15: 'Zinc (Zn)'
            },
            'units': {
                0: 'ug/L', 1: 'ug/L', 2: 'ug/L', 3: 'mg/L',
                4: 'CFU/100mL', 5: 'ug/L', 6: 'ug/L', 7: 'ug/L',
                8: 'mg/L', 9: 'mg/L', 10: 'mg/L', 11: 'mg/L',
                12: 'mg/L', 13: 'mg/L', 14: 'mg/L', 15: 'ug/L'
            }
        })
        self.wqs = samples.GrabSample(data, self.known_starttime,
                                           endtime=self.known_endtime,
                                           samplefreq=self.known_samplefreq,
                                           storm=None)
        self.wqs.siteid = self.known_siteid
        self.wqs.label = self.known_label
        self.wqs.wqstd = (
            info.wqstd_template()
                .assign(upper_limit=10)
                .query("season == 'summer'")
        )


class test_Storm(hydro_tests.test_Storm):
    def setup(self):
        # path stuff
        self.storm_file = resource_filename('wqio.data', 'teststorm_simple.csv')
        self.orig_record = pandas.read_csv(
            self.storm_file, index_col='date', parse_dates=True
        ).resample('5T').fillna(0)
        self.hr = wqio.HydroRecord(self.orig_record,
                                   precipcol='rain',
                                   inflowcol='influent',
                                   outflowcol='effluent',
                                   outputfreqMinutes=5,
                                   intereventHours=2,
                                   stormclass=samples.Storm)
        self.storm = samples.Storm(self.hr.data, 2,
                                   precipcol=self.hr.precipcol,
                                   inflowcol=self.hr.inflowcol,
                                   outflowcol=self.hr.outflowcol,
                                   freqMinutes=self.hr.outputfreq.n)

        self.known_columns = ['rain', 'influent', 'effluent', 'storm']
        self.known_index_type = pandas.DatetimeIndex
        self.known_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_end = pandas.Timestamp('2013-05-19 11:55')
        self.known_season = 'spring'
        self.known_precip_start = pandas.Timestamp('2013-05-19 06:10')
        self.known_precip_end = pandas.Timestamp('2013-05-19 09:10')
        self.known_inflow_start = pandas.Timestamp('2013-05-19 06:25')
        self.known_inflow_end = pandas.Timestamp('2013-05-19 11:30')
        self.known_outflow_start = pandas.Timestamp('2013-05-19 06:50')
        self.known_outflow_end = pandas.Timestamp('2013-05-19 11:50')
        self.known_duration_hours = 5.75
        self.known_antecedent_period_days = 0.17708333
        self.known_peak_precip_intensity = 1.32
        self.known_peak_inflow = 123.0
        self.known_peak_outflow = 41.0
        self.known_peak_precip_intensity_time = pandas.Timestamp('2013-05-19 08:00')
        self.known_peak_inflow_time = pandas.Timestamp('2013-05-19 08:35')
        self.known_peak_outflow_time = pandas.Timestamp('2013-05-19 08:45')
        self.known_centroid_precip = pandas.Timestamp('2013-05-19 07:41:58.705023')
        self.known_centroid_inflow = pandas.Timestamp('2013-05-19 08:44:31.252553')
        self.known_centroid_outflow = pandas.Timestamp('2013-05-19 08:59:51.675231')
        self.known_total_precip_depth = 1.39
        self.known_total_inflow_volume = 876600
        self.known_total_outflow_volume = 291900
        self.known_peak_lag_time = 0.75
        self.known_centroid_lag_time = 1.29804728

    def test_general_props(self):
        nt.assert_true(hasattr(self.storm, 'siteid'))
        self.storm.siteid = "test value"
        nt.assert_equal(self.storm.siteid, "test value")

        nt.assert_true(hasattr(self.storm, 'info'))
        self.storm.info = "test value"
        nt.assert_equal(self.storm.info, "test value")

    def test__hydrotable(self):
        nt.assert_true(hasattr(self.storm, '_hydro_table'))
        self.storm.info = {
            'duration_hours': self.known_duration_hours,
            'peak_outflow': self.known_peak_outflow,
            'peak_precip_intensity': self.known_peak_precip_intensity,
            'centroid_lag_hours': self.known_centroid_lag_time,
            'inflow_m3': self.known_total_inflow_volume,
            'outflow_m3': self.known_total_outflow_volume,
            'total_precip_depth': self.known_total_precip_depth,
        }

        known_table = (
            "Site,test\n"
            "Event Date,2013-05-19 06:10\n"
            "Antecedent Dry Period,0.2 days\n"
            "Event Duration,5.8 hr\n"
            "Peak Effluent Flow,41.0 L/s\n"
            "Peak Precipitation Intensity,1 mm/hr\n"
            "Lag Time,1.3 hr\n"
            "Estimated Total Influent Volume,876600 L\n"
            "Total Effluent Volume,291900 L\n"
            "Total Precipitation,1.4 mm\n"
        )
        wqio.testing.assert_bigstring_equal(
            self.storm._hydro_table('test'),
            known_table,
            "result_hydrotable.csv",
            "known_hydrotable.csv",
        )
