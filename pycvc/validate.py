
def groupby_col(groupby_col):
    valid_groups = ['season', 'grouped_season', 'year', 'storm_bin', 'samplestart']
    if groupby_col is None:
        return groupby_col
    elif groupby_col.lower() in valid_groups:
        return groupby_col.lower()
    else:
        raise ValueError("{} is not a valid time group ({})".format(groupby_col, valid_groups))


def sampletype(sampletype):
    """ Confirms that a given value is a valid sampletype and returns
    the all lowercase version of it.
    """
    if sampletype.lower() not in ('grab', 'composite'):
        raise ValueError("`sampletype` must be 'composite' or 'grab'")

    return sampletype.lower()


def rescol(rescol):
    """ Comfirms that a give value is a valid results column and returns
    the corresponding units column and results column.
    """
    if rescol.lower() == 'concentration':
        unitscol = 'units'
    elif rescol.lower() == 'load_outflow':
        unitscol = 'load_units'
    else:
        raise ValueError("`rescol` must be in ['concentration', 'load_outflow']")
    return rescol.lower(), unitscol
