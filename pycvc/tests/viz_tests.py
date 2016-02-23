from functools import partial
from pkg_resources import resource_filename
import os

import numpy as np
import pandas

from unittest import mock
import nose.tools as nt
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from matplotlib.testing.decorators import image_comparison, cleanup
import seaborn

from pycvc import viz


class Test_savefig(object):
    def setup(self):
        self.fig = pyplot.Figure()

    @cleanup
    def teardown(self):
        pyplot.close(self.fig)

    def test_png(self):
        with mock.patch.object(self.fig, 'savefig') as _save:
            viz.savefig(self.fig, 'test_png')
            _save.assert_called_once_with(
                os.path.join('output', 'img', 'test_png.png'),
                dpi=300, transparent=True, bbox_inches='tight'
            )

    def test_pdf(self):
        with mock.patch.object(self.fig, 'savefig') as _save:
            viz.savefig(self.fig, 'test_pdf', asPNG=False, asPDF=True)
            _save.assert_called_once_with(
                os.path.join('output', 'img', 'test_pdf.pdf'),
                dpi=300, transparent=True, bbox_inches='tight'
            )

    def test_extra_load(self):
        with mock.patch.object(self.fig, 'savefig') as _save:
            viz.savefig(self.fig, 'test_extra', extra='subdir', load=True)
            _save.assert_called_once_with(
                os.path.join('output', 'img', 'subdir', 'test_extra-load.png'),
                dpi=300, transparent=True, bbox_inches='tight'
            )


@nt.nottest
def load_test_data(filename, **opts):
    return pandas.read_csv(resource_filename('pycvc.tests.testdata', filename), **opts)


@image_comparison(baseline_images=['test_reduction_plot'], extensions=['png'])
def test_reduction_plot():
    red = load_test_data('test_reduction.csv')
    red.sort_values(by=['site', 'parameter'])
    fig = viz.reduction_plot(red, ['Cadmium (Cd)', 'Copper (Cu)'], 'parameter',
                             'site', 'season', (0.55, 0.03),
                             reduction='load_red',
                             lower='load_red_lower',
                             upper='load_red_upper')


@image_comparison(
    baseline_images=[
        'test_hydro_pairplot_ED1_season',
        'test_hydro_pairplot_LV2_year',
        'test_hydro_pairplot_LV4_outflow',
        'test_hydro_pairplot_ED1_grouped_season',
    ],
    extensions=['png']
)
def test_hydro_pairplot():
    hydro = load_test_data('test_hydro.csv', parse_dates=['start_date', 'end_date'])
    fig1 = viz.hydro_pairplot(hydro, 'ED-1', by='season', save=False).fig
    fig2 = viz.hydro_pairplot(hydro, 'LV-2', by='year', save=False).fig
    fig3 = viz.hydro_pairplot(hydro, 'LV-4', by='outflow', save=False).fig
    fig4 = viz.hydro_pairplot(hydro, 'ED-1', by='grouped_season', save=False).fig


@image_comparison(
    baseline_images=[
        'test_hydro_histogram_site_hasoutflow',
        'test_hydro_histogram_site_year',
        'test_hydro_histogram_single_site',
    ],
    extensions=['png']
)
def test_hydro_histogram():
    hydro = load_test_data('test_hydro.csv', parse_dates=['start_date', 'end_date'])
    viz.hydro_histogram(hydro, hue='site', row='has_outflow', save=False).fig
    viz.hydro_histogram(hydro, col='site', hue='year', col_wrap=2, save=False).fig
    viz.hydro_histogram(hydro.query("site == 'ED-1'"), palette='Blues', save=False).fig


@image_comparison(
    baseline_images=[
        'test_hydro_jointplot_depth_outflow',
        'test_hydro_jointplot_antecdays_outflow',
    ],
    extensions=['png']
)
def test_hydro_jointplot():
    hydro = load_test_data('test_hydro.csv', parse_dates=['start_date', 'end_date'])
    viz.hydro_jointplot(
        hydro=hydro, site='ED-1',
        xcol='total_precip_depth',
        ycol='outflow_mm',
        one2one=True,
        color='b',
        save=False
    )

    viz.hydro_jointplot(
        hydro=hydro, site='LV-4',
        xcol='antecedent_days',
        ycol='outflow_mm',
        conditions="outflow_mm > 0",
        one2one=False,
        color='g',
        save=False
    )


@image_comparison(baseline_images=['test_external_boxplot'], extensions=['png'])
def test_external_boxplot():
    tidy = load_test_data('external_tidy.csv')
    bmps = [
        'Bioretention', 'Detention Basin',
        'Manufactured Device', 'Retention Pond',
        'Wetland Channel',
    ]
    sites = ['ED-1', 'LV-1', 'LV-2', 'LV-4']
    params = ['Cadmium (Cd)', 'Copper (Cu)', 'Lead (Pb)', 'Zinc (Zn)']
    fg = viz.external_boxplot(tidy, categories=bmps, sites=sites, params=params,
                              units='ug/L')


@image_comparison(baseline_images=['test_seasonal_boxplot'], extensions=['png'])
def test_seasonal_boxplot():
    tidy = load_test_data('test_wq.csv')
    sites = ['ED-1', 'LV-1', 'LV-2', 'LV-4']
    params = ['Cadmium (Cd)', 'Total Suspended Solids']
    fg = viz.seasonal_boxplot(tidy, 'concentration', params=params, units='ug/L')
