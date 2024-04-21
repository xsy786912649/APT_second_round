import random
import numpy as np
import pickle
from pomdp import *
from model import *
from model_phase2 import * 
import sys
import time
import queue
import copy

with open(f'APT_data/hop.pickle','rb') as f:
    P0=pickle.load(f)
N_hop=[[] for i in range(7)]
for index_machine in list(P0):
    N_hop[int(P0[index_machine])].append(index_machine)
target=N_hop[0]
hop_1=N_hop[1]
hop_2=N_hop[2]
hop_3=N_hop[3]

def estimate_state(machine_state_list_belief_prability,cred_state_list_belief_prability):
    
    machine_state_list_estimated= [probability>0.81 for probability in machine_state_list_belief_prability]
    cred_state_list_estimated=[probability>0.81 for probability in cred_state_list_belief_prability]
    
    return machine_state_list_estimated,cred_state_list_estimated

def naive_estimate_state(naive_machine_state_list_estimated,observa_true):
    
    for i in range(len(naive_machine_state_list_estimated)):
        if i in observa_true:
            naive_machine_state_list_estimated[i]=True
    
    return naive_machine_state_list_estimated

def belief_state_update(my_pomdp_tem,machine_state_list_belief_prability,cred_state_list_belief_prability,action_contain_list,observation_machine,action_observation_list,observa_true):
    aaa=time.time()
    sampled_number=100
    machine_state_list_belief_prability_new=np.zeros_like(machine_state_list_belief_prability)
    cred_state_list_belief_prability_new=np.zeros_like(cred_state_list_belief_prability)
    i=0
    while i<sampled_number:
        machine_state_list_sampled= [np.random.rand(1)<probablity for probablity in machine_state_list_belief_prability]
        cred_state_list_sampled = [np.random.rand(1)<probablity for probablity in cred_state_list_belief_prability]
        machine_state_list_new_sampled,cred_state_list_new_sampled=my_pomdp_tem.state_transition_temp(machine_state_list_sampled,cred_state_list_sampled,action_contain_list)
        if all([machine_state_list_new_sampled[action_observation_list[i]]==observation_machine[i] for i in range(len(action_observation_list))]) :
            machine_state_list_belief_prability_new+=np.array([1.0/sampled_number*int(ele) for ele in machine_state_list_new_sampled])
            cred_state_list_belief_prability_new+=np.array([1.0/sampled_number*int(ele) for ele in cred_state_list_new_sampled])
            i=i+1

    for i in range(len(machine_state_list_belief_prability_new)):
        if i in observa_true:
            machine_state_list_belief_prability_new[i]=1.0

        elif machine_index_to_name(i) in hop_1+hop_2+hop_3: 
            if machine_state_list_belief_prability_new[i]<0.01:
                machine_state_list_belief_prability_new[i]=0.01
            elif machine_state_list_belief_prability_new[i]>0.8:
                machine_state_list_belief_prability_new[i]=0.8

        else: 
            if machine_state_list_belief_prability_new[i]<0.01:
                machine_state_list_belief_prability_new[i]=0.01
            elif machine_state_list_belief_prability_new[i]>0.05:
                machine_state_list_belief_prability_new[i]=0.05
    
    bbb=time.time()-aaa

    return machine_state_list_belief_prability_new, cred_state_list_belief_prability_new, bbb


def belief_state_update_delay(my_pomdp_tem,machine_state_list_belief_prability,cred_state_list_belief_prability,action_contain_list,observation_machine=None,action_observation_list=None,observa_true=None):
    if observation_machine==None:
        return machine_state_list_belief_prability, cred_state_list_belief_prability, 0
    
    aaa=time.time()
    sampled_number=100
    machine_state_list_belief_prability_new=np.zeros_like(machine_state_list_belief_prability)
    cred_state_list_belief_prability_new=np.zeros_like(cred_state_list_belief_prability)
    i_sample=0
    while i_sample<sampled_number:
        machine_state_list_sampled= [np.random.rand(1)<probablity for probablity in machine_state_list_belief_prability]
        cred_state_list_sampled = [np.random.rand(1)<probablity for probablity in cred_state_list_belief_prability]
        machine_state_list_new_sampled,cred_state_list_new_sampled=my_pomdp_tem.state_transition_temp(machine_state_list_sampled,cred_state_list_sampled,action_contain_list)
        
        gogogo=[]
        for iiiiii in range(len(action_observation_list)): 
            if observation_machine[iiiiii]==True:
                gogogo.append(machine_state_list_new_sampled[action_observation_list[iiiiii]]==observation_machine[iiiiii])
        if all(gogogo):
            machine_state_list_belief_prability_new+=np.array([1.0/sampled_number*int(ele) for ele in machine_state_list_new_sampled])
            cred_state_list_belief_prability_new+=np.array([1.0/sampled_number*int(ele) for ele in cred_state_list_new_sampled])
            i_sample=i_sample+1

    for i in range(len(machine_state_list_belief_prability_new)):
        if i in observa_true:
            machine_state_list_belief_prability_new[i]=1.0

        elif machine_index_to_name(i) in hop_1+hop_2+hop_3: 
            if machine_state_list_belief_prability_new[i]<0.01:
                machine_state_list_belief_prability_new[i]=0.01
            elif machine_state_list_belief_prability_new[i]>0.8 and machine_state_list_belief_prability_new[i]<0.9:
                machine_state_list_belief_prability_new[i]=0.8
            elif machine_state_list_belief_prability_new[i]>=0.9:
                machine_state_list_belief_prability_new[i]=0.9

        else: 
            if machine_state_list_belief_prability_new[i]<0.01:
                machine_state_list_belief_prability_new[i]=0.01
            elif machine_state_list_belief_prability_new[i]>0.05:
                machine_state_list_belief_prability_new[i]=0.05
    
    bbb=time.time()-aaa

    return machine_state_list_belief_prability_new, cred_state_list_belief_prability_new, bbb

def generate_siem(machine_state_list,cred_state_list):
    machine_state_prabablity =np.array(list(map(float, machine_state_list)))
    cred_state_prabablity =np.array(list(map(float, cred_state_list)))
    

    machine_state_prabablity=machine_state_prabablity+np.random.normal(0.0,0.2,len(machine_state_prabablity))
    cred_state_prabablity=cred_state_prabablity+np.random.normal(0.0,0.2,len(cred_state_prabablity))

    machine_state_prabablity[machine_state_prabablity<0.01]=0.01
    machine_state_prabablity[machine_state_prabablity>0.99]=0.99
    cred_state_prabablity[cred_state_prabablity<0.01]=0.01
    cred_state_prabablity[cred_state_prabablity>0.99]=0.99
    
    return machine_state_prabablity,cred_state_prabablity

def merge_belief(machine_state_list_belief_prability_delayed,cred_state_list_belief_prability_delayed,machine_state_list,cred_state_list):
    machine_state_list_belief_prability_delayed=copy.deepcopy(machine_state_list_belief_prability_delayed)
    cred_state_list_belief_prability_delayed=copy.deepcopy(cred_state_list_belief_prability_delayed)

    machine_state_siem,cred_state_siem=generate_siem(machine_state_list,cred_state_list) 

    machine_state_prabablity=machine_state_list_belief_prability_delayed*0.5+machine_state_siem*0.5
    machine_state_prabablity[machine_state_siem>0.9]=0.9
    machine_state_prabablity[machine_state_list_belief_prability_delayed>0.99]=1.0
    machine_state_prabablity[machine_state_siem<0.2]=0.02
    
    return machine_state_prabablity,cred_state_siem

if __name__ == "__main__":
    if len(sys.argv) != 2:
        weight=0.1
    else:
        weight = float(sys.argv[1]) #10.0
    with open(f"./model_"+str(weight)+".pkl",'rb') as f:
        value_map_dict=pickle.load(f)

    with open(f"./model_phase2_"+str(weight)+".pkl",'rb') as f:
        value_map_dict_further=pickle.load(f)

    average_number=0
    times=0
    total_iteration=0
    total_containing_number=0
    result = {}
    estimate_time=0

    nodelay_estimate_high_error_lists=[]
    nodelay_estimate_high_wrong=0
    nodelay_estimate_high_right=0
    nodelay_estimate_high_error=0
    
    naive_nodelay_estimate_high_error_lists=[]
    naive_nodelay_estimate_high_wrong=0
    naive_nodelay_estimate_high_right=0
    naive_nodelay_estimate_high_error=0

    delay_estimate_high_error_lists=[]
    delay_estimate_high_wrong=0
    delay_estimate_high_right=0
    delay_estimate_high_error=0

    nosiem_delay_estimate_high_error_lists=[]
    nosiem_delay_estimate_high_wrong=0
    nosiem_delay_estimate_high_right=0
    nosiem_delay_estimate_high_error=0

    for q in range(1):
        print("--------------------") 
        print(q)

        my_pomdp=POMDP()
        machine_state_list,cred_state_list,machine_state_list_belief_prability,cred_state_list_belief_prability=random_attacker_start(my_pomdp,seed=q)
        naive_machine_state_list_estimated=[False for i in range(len(machine_state_list))]

        machine_state_list_belief_prability_delayed=copy.deepcopy(machine_state_list_belief_prability)
        cred_state_list_belief_prability_delayed=copy.deepcopy(cred_state_list_belief_prability)

        machine_state_list_belief_prability_nosiem_delayed=copy.deepcopy(machine_state_list_belief_prability)
        cred_state_list_belief_prability_nosiem_delayed=copy.deepcopy(cred_state_list_belief_prability)

        observation_true_list=[]
        result[q] = [99999999999, -1]

        nodelay_estimate_high_error_this=[]
        naive_nodelay_estimate_high_error_this=[]

        oboservation_list_delay_queue=queue.Queue()

        for i in range(5000):
            machine_state_list_estimated,_=estimate_state(machine_state_list_belief_prability,cred_state_list_belief_prability)
            higher_state_current_machine=full_state_to_higher_state(machine_state_list_estimated) 
            current_valuedic_key=higher_state_to_valuedic_key(higher_state_current_machine) 

            if current_valuedic_key in list(value_map_dict_further): 
                Q_value_current=value_map_dict_further[current_valuedic_key] 
            else:
                simplest_state_current_machine=full_state_to_simplest_state(machine_state_list_estimated)
                current_valuedic_key=simplest_state_to_valuedic_key(simplest_state_current_machine)
                Q_value_current=value_map_dict[current_valuedic_key]

            nodelay_estimate_high=np.array(list(map(int, full_state_to_higher_state(machine_state_list_estimated))))
            true_high=np.array(list(map(int,full_state_to_higher_state(machine_state_list))))
            print("nodelay_estimate error ",np.sum(np.abs(nodelay_estimate_high-true_high)))
            
            if np.sum(np.abs(nodelay_estimate_high-true_high))>0:
                nodelay_estimate_high_wrong=nodelay_estimate_high_wrong+1
                nodelay_estimate_high_error+=np.sum(np.abs(nodelay_estimate_high-true_high))
            else:
                nodelay_estimate_high_right=nodelay_estimate_high_right+1
            nodelay_estimate_high_error_this.append(np.sum(np.abs(nodelay_estimate_high-true_high)))
 
            naive_nodelay_estimate_high=np.array(list(map(int, full_state_to_higher_state(naive_machine_state_list_estimated))))
            print("naive_nodelay_estimate error ",np.sum(np.abs(naive_nodelay_estimate_high-true_high)))
            if np.sum(np.abs(naive_nodelay_estimate_high-true_high))>0:
                naive_nodelay_estimate_high_wrong=naive_nodelay_estimate_high_wrong+1
                naive_nodelay_estimate_high_error+=np.sum(np.abs(naive_nodelay_estimate_high-true_high))
            else:
                naive_nodelay_estimate_high_right=naive_nodelay_estimate_high_right+1
            naive_nodelay_estimate_high_error_this.append(np.sum(np.abs(naive_nodelay_estimate_high-true_high)))

            machine_state_list_delay_estimated,_=estimate_state(machine_state_list_belief_prability_delayed,cred_state_list_belief_prability_delayed)
            estimate_delay_high=np.array(list(map(int, full_state_to_higher_state(machine_state_list_delay_estimated))))
            print("estimate_delay error ",np.sum(np.abs(estimate_delay_high-true_high)))
            if np.sum(np.abs(estimate_delay_high-true_high))>0:
                delay_estimate_high_wrong=delay_estimate_high_wrong+1
                delay_estimate_high_error+=np.sum(np.abs(estimate_delay_high-true_high))
            else:
                delay_estimate_high_right=delay_estimate_high_right+1 

            machine_state_list_nosiem_delay_estimated,_=estimate_state(machine_state_list_belief_prability_nosiem_delayed,cred_state_list_belief_prability_nosiem_delayed)
            estimate_nosiem_delay_high=np.array(list(map(int, full_state_to_higher_state(machine_state_list_nosiem_delay_estimated))))
            print("estimate_nosiem_delay error ",np.sum(np.abs(estimate_nosiem_delay_high-true_high)))
            if np.sum(np.abs(estimate_nosiem_delay_high-true_high))>0:
                nosiem_delay_estimate_high_wrong=nosiem_delay_estimate_high_wrong+1
                nosiem_delay_estimate_high_error+=np.sum(np.abs(estimate_nosiem_delay_high-true_high))
            else:
                nosiem_delay_estimate_high_right=nosiem_delay_estimate_high_right+1

            action_index=Q_value_current.index(max(Q_value_current)) 
            #action_index=0
            action_contain_list=index_to_action(action_index)

            #state_transition
            machine_state_list_new,cred_state_list_new=my_pomdp.state_transition(machine_state_list,cred_state_list,action_contain_list)

            machine_state_list=machine_state_list_new
            cred_state_list=cred_state_list_new

            observation_list=[machine_name_to_index(ele) for ele in hop_1+hop_2+hop_3]
            action_observation_list=random.sample(observation_list,2)

            observation_machine=my_pomdp.state_observation(machine_state_list,action_observation_list) 
            #print(action_observation_list)
            #print(observation_machine)

            for qq in range(len(observation_machine)):
                if observation_machine[qq]==True:
                    observation_true_list.append(action_observation_list[qq])

            aaaaaaaaa=(copy.deepcopy(observation_machine),copy.deepcopy(action_observation_list),copy.deepcopy(observation_true_list))
            oboservation_list_delay_queue.put(aaaaaaaaa)

            print([machine_state_list_belief_prability[action_observation_list[i]] for i in range(len(action_observation_list))])

            my_pomdp_tem=POMDP()
            machine_state_list_belief_prability,cred_state_list_belief_prability,time_computation=belief_state_update(my_pomdp_tem,machine_state_list_belief_prability,cred_state_list_belief_prability,action_contain_list,observation_machine,action_observation_list,observation_true_list)
            naive_machine_state_list_estimated=naive_estimate_state(naive_machine_state_list_estimated,observation_true_list)
            
            if i>100:
                observation_machine_delay,action_observation_list_delay,observation_true_list_delay=oboservation_list_delay_queue.get() 
            else:
                observation_machine_delay=None
                action_observation_list_delay=None
                observation_true_list_delay=None 

            machine_state_list_belief_prability_delayed,cred_state_list_belief_prability_delayed,time_computation1=belief_state_update_delay(my_pomdp_tem,machine_state_list_belief_prability_delayed,cred_state_list_belief_prability_delayed,action_contain_list,observation_machine_delay,action_observation_list_delay,observation_true_list_delay)
            machine_state_list_belief_prability_delayed,cred_state_list_belief_prability_delayed=merge_belief(machine_state_list_belief_prability_delayed,cred_state_list_belief_prability_delayed,machine_state_list,cred_state_list)
            
            machine_state_list_belief_prability_nosiem_delayed,cred_state_list_belief_prability_nosiem_delayed,time_computation2=belief_state_update_delay(my_pomdp_tem,machine_state_list_belief_prability_nosiem_delayed,cred_state_list_belief_prability_nosiem_delayed,action_contain_list,observation_machine_delay,action_observation_list_delay,observation_true_list_delay)
            estimate_time=estimate_time+time_computation

            #print(machine_state_list_belief_prability)
            #print([machine_state_list_belief_prability[i] for i in range(len(machine_state_list_belief_prability)) if machine_index_to_name(i) in hop_1+hop_2+hop_3])
            #print(machine_state_list_belief_prability[action_observation_list[0]],machine_state_list_belief_prability[action_observation_list[1]])
            print("iteration: "+str(i))
            total_iteration=total_iteration+1
            total_containing_number+=len(action_contain_list)
            print("------------reminder_---------------")
            print(str(times)+"/"+str(q))
            if times>0:
                print(1.0*average_number/times)
            print("------------next____________")

            machine_has_compr=[index for index in range(len(machine_state_list_new)) if machine_state_list_new[index]==True] 
            machine_has_compr_hop=[my_pomdp.hop[machine_index_to_name(index)] for index in machine_has_compr]
            result[q][0] = min(result[q][0], min(machine_has_compr_hop)) 
            if 0 in machine_has_compr_hop:
                continue
        result[q][1] = i
        nodelay_estimate_high_error_lists.append(nodelay_estimate_high_error_this)
        naive_nodelay_estimate_high_error_lists.append(naive_nodelay_estimate_high_error_this)

    #print(nodelay_estimate_high_error_lists)
    print("nodelay_estimate_high_wrong ", nodelay_estimate_high_wrong)
    #print("nodelay_estimate_high_right ", nodelay_estimate_high_right)
    print("nodelay_estimate_high_error ", nodelay_estimate_high_error)

    #print(naive_nodelay_estimate_high_error_lists)
    print("naive_nodelay_estimate_high_wrong ", naive_nodelay_estimate_high_wrong)
    #print("naive_nodelay_estimate_high_right ", naive_nodelay_estimate_high_right)
    print("naive_nodelay_estimate_high_error ", naive_nodelay_estimate_high_error)

    #print(delay_estimate_high_error_lists)
    print("delay_estimate_high_wrong ", delay_estimate_high_wrong)
    #print("delay_estimate_high_right ", delay_estimate_high_right)
    print("delay_estimate_high_error ", delay_estimate_high_error)

    #print(nosiem_delay_estimate_high_error_lists)
    print("nosiem_delay_estimate_high_wrong ", nosiem_delay_estimate_high_wrong)
    #print("nosiem_delay_estimate_high_right ", nosiem_delay_estimate_high_right)
    print("nosiem_delay_estimate_high_error ", nosiem_delay_estimate_high_error)

    #print(estimate_time)