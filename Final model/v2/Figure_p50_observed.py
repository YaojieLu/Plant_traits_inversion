import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Aru = [-3.028333333, -3.028333333, -3.794854811, -3.3, -1.97, -1.97, -1.97,\
       -1.69, -1.641892351, ]
Bpa = [-2.1685, -2.1685, -2.337, -2.337, -2.337, -2.14550845, -1.596]
Pgr = [-1.52]
p50 = pd.DataFrame({'Species': ['Aru']*len(Aru)+['Bpa']*len(Bpa)+\
                    ['Pgr']*len(Pgr), 'P50': Aru+Bpa+Pgr})

#sns.catplot(x='Species', y='P50', jitter=False, data=p50)
sns.scatterplot(x='Species', y='P50', hue='Species', data=p50)
