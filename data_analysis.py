import sys
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
if(len(sys.argv) != 3):
    print("Expected input: data_file statistic")
    exit(1)
data = pd.read_pickle(sys.argv[1])
stat = sys.argv[2]
avg_data = data.drop(columns = ['Game', 'Team'])
avg_data = avg_data.rolling(3, closed='left').mean()
avg_data = avg_data.dropna()
data = data.loc[avg_data.index]
no_points_data = data[data['Total Points'] == 0]
no_points_avg_data = avg_data.loc[no_points_data.index]
has_points_data = data[data['Total Points'] > 0]
has_points_avg_data = avg_data.loc[has_points_data.index]
print('No Points '+stat+' Normal Test: '+str(stats.normaltest(no_points_avg_data[stat]).pvalue))
print('Has Points '+stat+' Normal Test: '+str(stats.normaltest(has_points_avg_data[stat]).pvalue))
print('Has/No Points '+stat+' variance test: '+str(stats.levene(no_points_avg_data[stat], has_points_avg_data[stat]).pvalue))
plt.hist(no_points_avg_data[stat])
plt.hist(has_points_avg_data[stat])
plt.savefig('fig.png')
ttest = stats.ttest_ind(no_points_avg_data[stat], has_points_avg_data[stat])
print(stat+' T test: '+str(ttest.pvalue))