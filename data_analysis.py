import sys
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
if(len(sys.argv) != 2):
    print("Expected input: data_file")
    exit(1)
data = pd.read_pickle(sys.argv[1])
avg_data = data.drop(columns = ['Game', 'Team'])
avg_data = avg_data.rolling(3, closed='left').mean()
avg_data = avg_data.dropna()
data = data.loc[avg_data.index]
no_points_data = data[data['Total Points'] == 0]
no_points_avg_data = avg_data.loc[no_points_data.index]
has_points_data = data[data['Total Points'] > 0]
has_points_avg_data = avg_data.loc[has_points_data.index]

tests = {'stat': [], 'has_points': [], 'no_points': [], 'variance': [], 'ttest': [], 'mannwhitneyu': []}
for stat in list(avg_data.columns):
    tests['stat'].append(stat)
    tests['has_points'].append(stats.normaltest(no_points_avg_data[stat]).pvalue)
    tests['no_points'].append(stats.normaltest(has_points_avg_data[stat]).pvalue)
    tests['variance'].append(stats.levene(no_points_avg_data[stat], has_points_avg_data[stat]).pvalue)
    plt.clf()
    plt.hist(no_points_avg_data[stat])
    plt.hist(has_points_avg_data[stat])
    plt.savefig('plots/'+stat+'.png')
    ttest = stats.ttest_ind(no_points_avg_data[stat], has_points_avg_data[stat])
    mwu = stats.mannwhitneyu(no_points_avg_data[stat], has_points_avg_data[stat])
    tests['ttest'].append(ttest.pvalue)
    tests['mannwhitneyu'].append(mwu.pvalue)
results = pd.DataFrame(data=tests)
results.to_csv('data/analysis_results.csv')