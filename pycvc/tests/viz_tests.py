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


