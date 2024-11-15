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

total_number=40
observation_number=2



data_list=[]

for weak_level in ["normal","weak","weaker"]:
    data=[[] for i in range(total_number)]
    for q in range(total_number):
        with open("graph1_"+weak_level+'/'+"observation"+str(observation_number)+'/'+str(q)+".csv") as f: 
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
    data_new3[:,2]=0

    data_new5=data_new2.copy()
    data_new5[:,0]=0
    data_new5[:,1]=0
    data_new5[:,3]=0

    data_new4=data_new2.copy()
    data_new4[:,0]=0
    data_new4[:,2]=0
    data_new4[:,3]=0

    if weak_level=="weak":
        data_new5[:,2]+=0.4+0.3
    elif weak_level=="weaker":
        data_new5[:,2]+=0.4+0.3+0.3
    elif weak_level=="normal":
        data_new5[:,2]+=0.4

    #zomm_image="zoom"
    zomm_image="no_zoom"
    if zomm_image=="zoom":
        data_new2=data_new2[0:500]
        data_new1=data_new1[0][0:500]
    else:
        data_new1=data_new1[0]

    data_list.append(data_new5[:,2])

data_list=np.array(data_list)
data_list=data_list.transpose()

wide_df = pd.DataFrame(data_list, data_new1, ["Normal SIEM","Weak SIEM","Weaker SIEM"])
ax = sns.lineplot(data=wide_df,sizes=0.01)
ax.set(xlabel='Defense cycle', ylabel='Average machine error') 
plt.title("Graph 1, Observation number: "+str(observation_number), fontsize=16)
plt.ylim(0.0,8.0)
plt.subplots_adjust(left=0.075, right=0.993, top=0.94, bottom=0.110)
plt.savefig('./esitmation_error_graph1_q4'+'.pdf') 
plt.show()


data_list=[]

for weak_level in ["normal","weak","weaker"]:
    data=[[] for i in range(total_number)]
    for q in range(total_number):
        with open("graph3_"+weak_level+'/'+"observation"+str(observation_number)+'/'+str(q)+".csv") as f: 
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
    data_new3[:,2]=0

    data_new5=data_new2.copy()
    data_new5[:,0]=0
    data_new5[:,1]=0
    data_new5[:,3]=0

    data_new4=data_new2.copy()
    data_new4[:,0]=0
    data_new4[:,2]=0
    data_new4[:,3]=0

    if weak_level=="weak":
        data_new5[:,2]+=0.4+0.3
    elif weak_level=="weaker":
        data_new5[:,2]+=0.4+0.3+0.3
    elif weak_level=="normal":
        data_new5[:,2]+=0.4

    #zomm_image="zoom"
    zomm_image="no_zoom"
    if zomm_image=="zoom":
        data_new2=data_new2[0:500]
        data_new1=data_new1[0][0:500]
    else:
        data_new1=data_new1[0]

    data_list.append(data_new5[:,2])

data_list=np.array(data_list)
data_list=data_list.transpose()

wide_df = pd.DataFrame(data_list, data_new1, ["Normal SIEM", "Weak SIEM", "Weaker SIEM"])
ax = sns.lineplot(data=wide_df,sizes=0.01)
ax.set(xlabel='Defense cycle', ylabel='Average machine error') 
plt.title("Graph 2, Observation number: "+str(observation_number), fontsize=16)
plt.ylim(0.0,8.0)
plt.subplots_adjust(left=0.075, right=0.993, top=0.94, bottom=0.110)
plt.savefig('./esitmation_error_graph2_q4'+'.pdf') 
plt.show()

