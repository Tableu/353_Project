import pandas as pd
import matplotlib.pyplot as plt
import sys
import random

data = pd.read_csv(sys.argv[1])
print(data)

cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.*i/19) for i in range(19)]

for i, txt in enumerate(data['Player']):
    plt.scatter(data['Prediction'][i],data['Yahoo Roster Percentage(%)'][i], c=colors[i], label=txt)
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
plt.annotate('W. Foegele', (3.477193, 1), xytext=(0,4),textcoords='offset points')
plt.annotate('C. McMichael', (2.765030, 1), xytext=(0,4),textcoords='offset points')
plt.annotate('M. Joseph', (2.878466, 9), xytext=(0,4),textcoords='offset points')
plt.annotate('C. Fischer', (3.325188, 0), xytext=(-25,4),textcoords='offset points')
plt.annotate('T. Bertuzzi', (2.990692, 26), xytext=(0,4),textcoords='offset points')
plt.annotate('T. Hertl', (3.234152, 37), xytext=(0,4),textcoords='offset points')
plt.xlabel('Predicted Shot Count')
plt.ylabel('Roster Percentage (%)')
plt.title('Top 19 players by predicted shot count')
plt.tight_layout()
plt.savefig('plots/top_rankings.png')