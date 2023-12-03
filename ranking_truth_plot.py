import pandas as pd
import matplotlib.pyplot as plt

# rankings = pd.read_csv('output/Player_rankings.csv')
# rankings = rankings.drop(rankings.index.to_list()[20:], axis=0)
# rankings['Actual_Shots'] = [2, 4, 3, 4, 2, 2, 2, 5, 8, 4, 2, 4, 2, 0, 2, 2, 0, 2, 3, 0]
rankings = pd.read_csv('output/ranking_analysis.csv')
plt.plot(rankings['Truth'], rankings['Prediction'], 'b.', alpha=0.5)
# plt.plot([3, 4, 5], [3, 4, 5], 'r-', linewidth=3)
# plt.plot()
plt.ylabel('Expected Shots')
plt.xlabel("Actual Shots")
plt.title('Actual vs. Expected Shots')
# plt.show()
plt.savefig('output/Actual_vs_expected_shots.png')
