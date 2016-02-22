import nose.tools as nt

from pycvc import validate


class test_rescol(object):
    def test_conc(self):
        rescol, unitscol = validate.rescol('concenTRATION')
        nt.assert_equal(rescol, 'concentration')
        nt.assert_equal(unitscol, 'units')

    def test_load(self):
        rescol, unitscol = validate.rescol('load_outflow')
        nt.assert_equal(rescol, 'load_outflow')
        nt.assert_equal(unitscol, 'load_units')

    @nt.raises(ValueError)
    def test_junk(self):
        validate.rescol('junk')


class test_groupby_col(object):
    def test_season(self):
        nt.assert_equal(validate.groupby_col('seASon'), 'season')

    def test_year(self):
        nt.assert_equal(validate.groupby_col('YEAR'), 'year')

    def test_grouped_season(self):
        nt.assert_equal(validate.groupby_col('grouped_season'), 'grouped_season')

    @nt.raises(ValueError)
    def test_junk(self):
        validate.groupby_col('junk')\


class test_sampletype(object):
    def test_grab(self):
        nt.assert_equal(validate.sampletype('GraB'), 'grab')

    def test_composite(self):
        nt.assert_equal(validate.sampletype('comPOSite'), 'composite')

    @nt.raises(ValueError)
    def test_junk(self):
        validate.sampletype('junk')

