import numpy as np
import pandas


def getPOCs():
    return [p['cvcname'] for p in POC_dicts]


def getPOCInfo(critcol, critval, valcol):
    values = list(filter(lambda poc: poc[critcol] == critval, POC_dicts))
    if len(values) > 1:
        raise ValueError('`getPOCInfo` found multiple records')
    else:
        return values[0][valcol]

def wqstd_template():
    seasons = ['summer', 'autumn', 'winter', 'spring']
    _template = pandas.DataFrame({
        'parameter': [p['cvcname'] for p in POC_dicts],
        'units': [p['conc_units']['plain'] for p in POC_dicts]
    })

    df = pandas.concat([_template.assign(season=s) for s in seasons])
    df['influent median'] = np.nan
    return df


LITERS_PER_CUBICMETER = 1.0e3
MICROGRAMS_PER_GRAM = 1.0e6
MILLIGRAMS_PER_GRAM = 1.0e3

POC_dicts = [
    {
        'cvcname': 'Aluminum (Al)',
        'bmpname': 'Aluminum, Total',
        'nsqdname': None,
        'conc_units': {
            'plain': 'ug/L',
            'tex': r'\si[per-mode=symbol]{\micro\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MICROGRAMS_PER_GRAM,
        'group': 'A',
        'include': False,
    }, {
        'cvcname': 'Cadmium (Cd)',
        'bmpname': 'Cadmium, Total',
        'nsqdname': 'Cadmium',
        'conc_units': {
            'plain': 'ug/L',
            'tex': r'\si[per-mode=symbol]{\micro\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MICROGRAMS_PER_GRAM,
        'group': 'A',
        'include': True,
    }, {
        'cvcname': 'Copper (Cu)',
        'bmpname': 'Copper, Total',
        'nsqdname': 'Copper',
        'conc_units': {
            'plain': 'ug/L',
            'tex': r'\si[per-mode=symbol]{\micro\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MICROGRAMS_PER_GRAM,
        'group': 'A',
        'include': True,
    }, {
        'cvcname': 'Iron (Fe)',
        'bmpname': 'Iron, Total',
        'nsqdname': 'Iron as Fe',
        'conc_units': {
            'plain': 'ug/L',
            'tex': r'\si[per-mode=symbol]{\micro\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MICROGRAMS_PER_GRAM,
        'group': 'A',
        'include': True,
    }, {
        'cvcname': 'Lead (Pb)',
        'bmpname': 'Lead, Total',
        'nsqdname': 'Lead',
        'conc_units': {
            'plain': 'ug/L',
            'tex': r'\si[per-mode=symbol]{\micro\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MICROGRAMS_PER_GRAM,
        'group': 'A',
        'include': True,
    }, {
        'cvcname': 'Nickel (Ni)',
        'bmpname': 'Nickel, Total',
        'nsqdname': 'Nickel',
        'conc_units': {
            'plain': 'ug/L',
            'tex': r'\si[per-mode=symbol]{\micro\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MICROGRAMS_PER_GRAM,
        'group': 'A',
        'include': True,
    }, {
        'cvcname': 'Zinc (Zn)',
        'bmpname': 'Zinc, Total',
        'nsqdname': 'Zinc',
        'conc_units': {
            'plain': 'ug/L',
            'tex': r'\si[per-mode=symbol]{\micro\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MICROGRAMS_PER_GRAM,
        'group': 'A',
        'include': True,
    }, {
        'cvcname': 'Dissolved Chloride (Cl)',
        'bmpname': 'Chloride, Dissolved',
        'nsqdname': 'Chloride',
        'conc_units': {
            'plain': 'mg/L',
            'tex': r'\si[per-mode=symbol]{\micro\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MILLIGRAMS_PER_GRAM,
        'group': 'B',
        'include': False,
    }, {
        'cvcname': 'Nitrate (N)',
        'bmpname': 'Nitrogen, Nitrate (NO3) as N',
        'nsqdname': 'Nitrate as N',
        'conc_units': {
            'plain': 'mg/L',
            'tex': r'\si[per-mode=symbol]{\milli\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MILLIGRAMS_PER_GRAM,
        'group': 'B',
        'include': True,
    }, {
        'cvcname': 'Nitrate + Nitrite',
        'bmpname': 'Nitrogen, Nitrite (NO2) + Nitrate (NO3) as N',
        'nsqdname': 'N02+NO3',
        'conc_units': {
            'plain': 'mg/L',
            'tex': r'\si[per-mode=symbol]{\micro\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MILLIGRAMS_PER_GRAM,
        'group': 'B',
        'include': True,
    }, {
        'cvcname': 'Total Kjeldahl Nitrogen (TKN)',
        'bmpname': 'Kjeldahl nitrogen (TKN)',
        'nsqdname': 'Total Kjeldahl Nitrogen',
        'conc_units': {
            'plain': 'mg/L',
            'tex': r'\si[per-mode=symbol]{\milli\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MILLIGRAMS_PER_GRAM,
        'group': 'B',
        'include': False,
    }, {
        'cvcname': 'Orthophosphate (P)',
        'bmpname': 'Phosphorus, orthophosphate as P',
        'nsqdname': 'Phosphate Ortho as P',
        'conc_units': {
            'plain': 'mg/L',
            'tex': r'\si[per-mode=symbol]{\milli\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MILLIGRAMS_PER_GRAM,
        'group': 'B',
        'include': False,
    }, {
        'cvcname': 'Total Phosphorus',
        'bmpname': 'Phosphorus as P, Total',
        'nsqdname': 'Phosphorous as P',
        'conc_units': {
            'plain': 'mg/L',
            'tex': r'\si[per-mode=symbol]{\milli\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MILLIGRAMS_PER_GRAM,
        'group': 'B',
        'include': True,
    }, {
        'cvcname': 'Escherichia coli',
        'bmpname': 'Escherichia coli',
        'nsqdname': 'E. Coli',
        'conc_units': {
            'plain': 'CFU/100 mL',
            'tex': r'CFU/100 mL'
        },
        'load_units': 'CFU',
        'load_factor': 10 * LITERS_PER_CUBICMETER,
        'group': 'B',
        'include': False,
    }, {
        'cvcname': 'Total Oil & Grease',
        'bmpname': None,
        'nsqdname': 'Oil and Grease',
        'conc_units': {
            'plain': 'mg/L',
            'tex': r'\si[per-mode=symbol]{\milli\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MILLIGRAMS_PER_GRAM,
        'group': 'B',
        'include': False,
    }, {
        'cvcname': 'Total Suspended Solids',
        'bmpname': 'Total suspended solids',
        'nsqdname': 'Total Suspended Solids',
        'conc_units': {
            'plain': 'mg/L',
            'tex': r'\si[per-mode=symbol]{\milli\gram\per\liter}'
        },
        'load_units': 'g',
        'load_factor': LITERS_PER_CUBICMETER / MILLIGRAMS_PER_GRAM,
        'group': 'A',
        'include': True,
    },
]
