import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
plt.style.use('ggplot')
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
plt.rcParams.update({'font.size': 14})

observation_number=1

data=[[],[],[],[],[],[],[],[],[],[]]
for q in range(10):
    with open("observation"+str(observation_number)+'/'+str(q)+".csv") as f: 
        spamreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in spamreader:
            data[q].append([float(x) for x in row])

data_new=np.array(data) 
data_new0=np.mean(data_new,axis=0)
data_new1=data_new0.transpose()

data_new2=data_new1[[1,3,4]]
data_new2=data_new2.transpose()

print(data_new2)

wide_df = pd.DataFrame(data_new2, data_new1[0], ["No_delayed_IDS", "Delayed_IDS_with_Siems","Delayed_IDS_without_Siems"])
ax = sns.lineplot(data=wide_df)
ax.set(xlabel='Defense cycle', ylabel='Average machine error') 
plt.subplots_adjust(left=0.046, right=0.982, top=0.982, bottom=0.114)
plt.savefig('./figures/esitmation_error_graph3'+"_observation"+str(observation_number)+'.pdf') 
plt.show()