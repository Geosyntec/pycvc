from functools import partial
from pkg_resources import resource_filename

import numpy as np
import pandas

import nose.tools as nt
from matplotlib.testing.decorators import image_comparison
import seaborn

from pycvc import viz


@nt.nottest
def load_test_data(filename, **opts):
    return pandas.read_csv(resource_filename('pycvc.tests.testdata', filename), **opts)


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
