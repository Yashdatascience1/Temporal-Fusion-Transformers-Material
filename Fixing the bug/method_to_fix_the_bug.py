jj = pandas_df[pandas_df.index.month.isin([6, 7])]
print(jj.groupby(jj.index.year).agg(
    GANGA    = ('GANGA_DUSSEHRA_DAYS', 'sum'),
    HARTALIK = ('HARTALIK_TEEJ_DAYS', 'sum'),
    NAG      = ('NAG_PANCHAMI_DAYS', 'sum'),
    MARRIAGE = ('MARRIAGE_DAYS', 'sum'),
    NET_SALES= ('NET_SALES', 'sum'),
))

import matplotlib.pyplot as plt

explainer.plot_variable_selection(result)

fig = plt.gcf()
fig.set_size_inches(18, 12)

for ax in fig.get_axes():
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
    ax.tick_params(axis='both', labelsize=11)
    ax.title.set_fontsize(13)

fig.tight_layout()
plt.show()