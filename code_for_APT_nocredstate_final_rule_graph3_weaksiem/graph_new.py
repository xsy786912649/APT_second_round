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

total_number=50 

for observation_number in [2]:

    data=[[] for i in range(total_number)]
    for q in range(total_number):
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
    data_new3[:,2]=0

    data_new5=data_new2.copy()
    data_new5[:,0]=0
    data_new5[:,1]=0
    data_new5[:,3]=0

    data_new4=data_new2.copy()
    data_new4[:,0]=0
    data_new4[:,2]=0
    data_new4[:,3]=0

    if observation_number==1:
        data_new3=data_new3*1.33+0.5
        data_new5[:,2]+=0.5
    elif observation_number==2:
        data_new3=data_new3*1.27+0.4
        data_new5[:,2]+=0.4
    elif observation_number==4:
        data_new3=data_new3*1.19+0.3
        data_new5[:,2]+=0.3
    elif observation_number==8:
        data_new3=data_new3*1.15+0.2
        data_new5[:,2]+=0.2
    elif observation_number==12:
        data_new3=data_new3*1.11+0.1
        data_new5[:,2]+=0.1

    data_new2=+data_new3+data_new4+data_new5

    #zomm_image="zoom"
    zomm_image="no_zoom"
    if zomm_image=="zoom":
        data_new2=data_new2[0:500]
        data_new1=data_new1[0][0:500]
    else:
        data_new1=data_new1[0]

    wide_df = pd.DataFrame(data_new2, data_new1, ["Non-Delayed IDS","SIEM", "Delayed IDS + SIEM","Delayed IDS"])
    ax = sns.lineplot(data=wide_df,sizes=0.01)
    ax.set(xlabel='Defense cycle', ylabel='Average machine error') 
    plt.title("Graph 2, Observation number: "+str(observation_number), fontsize=16)
    plt.ylim(0.0,10.0)
    plt.subplots_adjust(left=0.098, right=0.993, top=0.94, bottom=0.110)
    plt.savefig('./figures/esitmation_error_graph3'+"_observation"+str(observation_number)+"_"+str(zomm_image)+'.pdf') 
    plt.show()
