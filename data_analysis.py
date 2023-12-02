import sys
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
if(len(sys.argv) != 2):
    print("Expected input: data_file")
    exit(1)
data = pd.read_pickle(sys.argv[1])
avg_data = data.drop(columns = ['Game', 'Team'])
avg_data = avg_data.groupby('Id', as_index=False).rolling(3, closed='left').mean()
avg_data = avg_data.dropna()
avg_data = avg_data.drop(columns=['Id'])
data = data.loc[avg_data.index]
no_points_data = data[data['Total Points'] == 0]
no_points_avg_data = avg_data.loc[no_points_data.index]
has_points_data = data[data['Total Points'] > 0]
has_points_avg_data = avg_data.loc[has_points_data.index]

tests = {'stat': [], 'has_points normal test': [], 'no_points normal test': [], 'variance test': [], 'ttest': [], 'mannwhitneyu': []}
for stat in list(avg_data.columns):
    tests['stat'].append(stat)
    tests['has_points normal test'].append(stats.normaltest(no_points_avg_data[stat]).pvalue)
    tests['no_points normal test'].append(stats.normaltest(has_points_avg_data[stat]).pvalue)
    tests['variance test'].append(stats.levene(no_points_avg_data[stat], has_points_avg_data[stat]).pvalue)
    plt.clf()
    plt.hist(no_points_avg_data[stat])
    plt.hist(has_points_avg_data[stat])
    plt.savefig('plots/'+stat+'.png')
    plt.close()
    ttest = stats.ttest_ind(no_points_avg_data[stat], has_points_avg_data[stat])
    mwu = stats.mannwhitneyu(no_points_avg_data[stat], has_points_avg_data[stat])
    tests['ttest'].append(ttest.pvalue)
    tests['mannwhitneyu'].append(mwu.pvalue)
results = pd.DataFrame(data=tests)
results = results.sort_values('stat')
results.to_csv('data/analysis_results.csv')

# correlation stats
output = 'data/moderately_strong_correlation_stats.txt'
current_season = pd.read_pickle('data/current_season_game_logs.pkl')
last_season = pd.read_pickle('data/last_season_game_logs.pkl')
all_logs = pd.concat([current_season, last_season])
all_logs = all_logs.reset_index()
all_logs = all_logs.drop(['Game', 'Team', 'Id', 'Date'], axis=1)

X = all_logs.drop('Shots', axis=1)
y = all_logs['Total Points']
y.groupby(y).count().to_csv('data/Points_distribution')
plt.hist(y)
plt.title("Distribution of Points")
plt.xlabel('Points')
plt.ylabel('Frequency')
plt.savefig('plots/Points_Histogram.png')
y = all_logs['Shots']

# plt.figure()
with open(output, 'w') as f:
    for column in X.columns:
        plt.figure()
        plt.scatter(X[column], y)
        fit = stats.linregress(X[column], y)
        plt.plot(X[column], ((X[column]*fit.slope) + fit.intercept), 'r-', linewidth=3)
        plt.xlabel(column)
        plt.ylabel('shots')
        # plt.savefig('plots/'+ str(column) + '_corr' + '.png')
        if(fit.rvalue > 0.5):
            f.write(str(column) + ' '+ str(fit.rvalue) + '\n')
            plt.savefig('plots/corr_plots/'+ str(column) + '_corr' + '.png')
            # plt.show()
        plt.close()
    f.close()