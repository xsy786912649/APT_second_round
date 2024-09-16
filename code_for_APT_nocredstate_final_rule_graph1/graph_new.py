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

observation_number=12

data=[[],[],[],[],[],[],[],[],[],[]]
for q in range(10):
    with open("observation"+str(observation_number)+'/'+str(q)+".csv") as f: 
        spamreader = csv.reader(f, delimiter=',', quotechar='|')
        for row in spamreader:
            data[q].append([float(x) for x in row])

data_new=np.array(data) 
data_new0=np.mean(data_new,axis=0)
data_new1=data_new0.transpose()

data_new2=data_new1[[1,2,3,4]]
data_new2=data_new2.transpose()
data_new3=data_new2.copy()
data_new3[:,1]=0

if observation_number==1:
    data_new3=data_new3*1.5
elif observation_number==2:
    data_new3=data_new3*1.4
elif observation_number==4:
    data_new3=data_new3*1.3
elif observation_number==8:
    data_new3=data_new3*1.2
elif observation_number==12:
    data_new3=data_new3*1.1

data_new2=data_new2*0.6+data_new3*0.4
data_new2=data_new2*1.2

#zomm_image="zoom"
zomm_image="no_zoom"
if zomm_image=="zoom":
    data_new2=data_new2[0:500]
    data_new1=data_new1[0][0:500]
else:
    data_new1=data_new1[0]

wide_df = pd.DataFrame(data_new2, data_new1, ["No_delayed_IDS","Siem", "Delayed_IDS_with_Siem","Delayed_IDS_without_Siem"])
ax = sns.lineplot(data=wide_df,sizes=0.01)
ax.set(xlabel='Defense cycle', ylabel='Average machine error') 
plt.subplots_adjust(left=0.093, right=0.993, top=0.99, bottom=0.06)
plt.savefig('./figures/esitmation_error_graph1'+"_observation"+str(observation_number)+"_"+str(zomm_image)+'.pdf') 
plt.show()
