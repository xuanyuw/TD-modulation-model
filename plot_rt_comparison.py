import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pickle import load
import pandas as pd
from os.path import join

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

f_dirs = [
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_highTestCoh_model",
    "crossOutput_noInterneuron_noMTConn_gaussianInOut_WeightLambda1_shufFeedback_model"
]

model_types = [f_dir.split('_')[-2] for f_dir in f_dirs]

df = pd.DataFrame(columns=['model_type', 'rt'])
for i in range(len(f_dirs)):
    rt = load(open(join(f_dirs[i], '%s_all_rt.pkl'%model_types[i]), 'rb'))
    temp_df = pd.DataFrame({'rt': rt, 'model_type':[model_types[i]]*len(rt)})
    df = df.append(temp_df, ignore_index=True)

sns.boxplot(data=df, x='rt', y='model_type')
plt.show()