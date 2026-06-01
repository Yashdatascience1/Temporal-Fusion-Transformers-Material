jj = pandas_df[pandas_df.index.month.isin([6, 7])]
print(jj.groupby(jj.index.year).agg(
    GANGA    = ('GANGA_DUSSEHRA_DAYS', 'sum'),
    HARTALIK = ('HARTALIK_TEEJ_DAYS', 'sum'),
    NAG      = ('NAG_PANCHAMI_DAYS', 'sum'),
    MARRIAGE = ('MARRIAGE_DAYS', 'sum'),
    NET_SALES= ('NET_SALES', 'sum'),
))