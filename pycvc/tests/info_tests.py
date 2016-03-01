import nose.tools as nt
import pandas.util.testing as pdtest

import pandas

from pycvc import info


def test_constants():
    nt.assert_equal(info.LITERS_PER_CUBICMETER, 1000)
    nt.assert_equal(info.MICROGRAMS_PER_GRAM, 1000000)
    nt.assert_equal(info.MILLIGRAMS_PER_GRAM, 1000)


def test_POC_dicts():
    for poc in info.POC_dicts:
        expected_keys = sorted([
            'cvcname', 'bmpname', 'nsqdname',
            'conc_units', 'load_units', 'load_factor',
            'group', 'include'
        ])
        keys = sorted(list(poc.keys()))
        nt.assert_list_equal(keys, expected_keys)
        nt.assert_true(poc['group'] in ['A', 'B'])
        nt.assert_true(poc['include'] in [True, False])
        nt.assert_true(poc['conc_units']['plain'] in ['ug/L', 'mg/L', 'CFU/100 mL'])


def test_getPOCs():
    nt.assert_true(isinstance(info.getPOCs(), list))


def test_getPOCInfo():
    nt.assert_equal(
        info.getPOCInfo('nsqdname', 'Copper', 'cvcname'),
        'Copper (Cu)'
    )


@nt.raises(ValueError)
def test_getPOCInfo_non_unique_result():
    info.getPOCInfo('group', 'A', 'cvcname')


def test_wqstd_template():
    std = info.wqstd_template()
    nt.assert_list_equal(std.columns.tolist(), ['parameter', 'units', 'season', 'influent median'])
    expected_shape = (16*4, 4) #(POCs x seasons, cols)
    nt.assert_tuple_equal(std.shape, expected_shape)



