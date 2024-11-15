import numpy as np
import pandas as pd

rule_groups = [
        [2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509],
        [5300, 5301, 5302, 5303, 5304, 5305, 5306],
        [5901, 5902, 5903, 5904, 5905],
        [5400, 5401, 5402, 5407, 5403, 5404, 5405, 5406],
        [9100, 9101, 9102],
        [5600, 5601, 5602, 5603, 5604, 5631],
        [92020, 92022, 92026, 92030, 92031, 92033, 92034, 92035, 92037], 
        [92038, 92039, 92040, 92046, 92055, 92056, 92062, 92080, 92023, 92024]
    ]

def run_simulation_compromised(severity_growth_factors, num_events=100, min_fn_level=0):
    # Define the rule data
    rules_data = {
        'Rule ID': [
            2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509,
            5300, 5301, 5302, 5303, 5304, 5305, 5306,
            5901, 5902, 5903, 5904, 5905,
            5400, 5401, 5402, 5407, 5403, 5404, 5405, 5406,
            9100, 9101, 9102,
            5600, 5601, 5602, 5603, 5604, 5631,
            92020, 92022, 92026, 92030, 92031, 92033, 92034, 92035,
            92037, 92038, 92039, 92040, 92046, 92055, 92056, 92062,
            92080, 92023, 92024
        ],
        'Severity Level': [
            6, 11, 6, 10, 4, 4, 4, 4, 6,
            1, 6, 10, 4, 4, 5, 1,
            9, 9, 4, 9, 1,
            1, 6, 4, 4, 5, 11, 6, 6,
            1, 1, 1,
            1, 6, 4, 6, 6, 11,
            7, 4, 15, 4, 4, 4, 4, 4, 4, 13, 4, 13, 13, 13, 15, 15, 8, 9, 15
        ]
    }

    # Define rule groups (each group as a list)
    rule_groups = [
        [2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509],
        [5300, 5301, 5302, 5303, 5304, 5305, 5306],
        [5901, 5902, 5903, 5904, 5905],
        [5400, 5401, 5402, 5407, 5403, 5404, 5405, 5406],
        [9100, 9101, 9102],
        [5600, 5601, 5602, 5603, 5604, 5631],
        [92020, 92022, 92026, 92030, 92031, 92033, 92034, 92035, 92037], 
        [92038, 92039, 92040, 92046, 92055, 92056, 92062, 92080, 92023, 92024]
    ]

    # Create a mapping of each rule to its group
    rule_to_group = {}
    for group in rule_groups:
        for rule in group:
            rule_to_group[rule] = group

    # Create DataFrame
    df_rules = pd.DataFrame(rules_data)

    # Function to map severity level to its class
    def get_severity_class(severity_level):
        if 1 <= severity_level <= 5:
            return 1
        elif 6 <= severity_level <= 10:
            return 2
        elif 11 <= severity_level <= 15:
            return 3
        else:
            return None

    # Assign the user-selected growth factors to the corresponding rules
    df_rules['Severity Growth Factor'] = df_rules['Severity Level'].apply(
        lambda severity: severity_growth_factors[get_severity_class(severity)] if get_severity_class(severity) else 0
    )

    # Define a function to calculate FN using the formula
    def calculate_fn(severity_level, severity_growth_factor, min_fn_level=0):
        return min_fn_level + (1 - min_fn_level) * (1 - np.exp(-(severity_growth_factor) * severity_level))

    # Apply the FN formula to each rule in the DataFrame
    df_rules['FN'] = df_rules.apply(lambda x: calculate_fn(x['Severity Level'], x['Severity Growth Factor'], min_fn_level), axis=1)

    # Normalize probabilities (as in previous code)
    A = 1  # scaling factor
    B = 1.1  # logistic growth rate (as given)
    C = 6  # midpoint of the logistic function (as given)

    # Calculate logistic frequencies using the formula: A / (1 + exp(B * (severity_level - C)))
    df_rules['Probability'] = A / (1 + np.exp(B * (df_rules['Severity Level'] + 1 - C)))
    df_rules['Normalized Probability'] = df_rules['Probability'] / df_rules['Probability'].sum()

    # Initialize counters for rule selection
    df_rules['Alert Count'] = 0
    df_rules['Non-Alert Count'] = 0
    df_rules['Total Selected'] = 0

    # Simulate the rule selection process
    for _ in range(num_events):
        # Randomly choose one rule based on probability
        selected_rule = np.random.choice(df_rules['Rule ID'], p=df_rules['Normalized Probability'])

        # Get the group of the selected rule
        group = rule_to_group[selected_rule]

        # Select all rules in the group
        for rule_id in group:
            fn_value = df_rules.loc[df_rules['Rule ID'] == rule_id, 'FN'].values[0]
            df_rules.loc[df_rules['Rule ID'] == rule_id, 'Total Selected'] += 1

            # Check if the rule triggers an alert or not based on FN
            if np.random.rand() > fn_value:
                df_rules.loc[df_rules['Rule ID'] == rule_id, 'Alert Count'] += 1
            else:
                df_rules.loc[df_rules['Rule ID'] == rule_id, 'Non-Alert Count'] += 1

    # Calculate a1c1 (alert ratio) and a0c1 (non-alert ratio)
    df_rules['a1c1'] = df_rules['Alert Count'] / (df_rules['Alert Count']+ df_rules['Non-Alert Count'])
    df_rules['a0c1'] = df_rules['Non-Alert Count'] / (df_rules['Alert Count']+ df_rules['Non-Alert Count'])
    df_rules['select_probablity'] = df_rules['Total Selected'] / float(num_events)

    # Replace NaN values with 0 (in case there are rules that were never selected)
    df_rules['a1c1'] = df_rules['a1c1'].fillna(0)
    df_rules['a0c1'] = df_rules['a0c1'].fillna(0)

    # Return the DataFrame with the selected columns for display
    return df_rules[['Rule ID', 'Severity Level', 'Total Selected', 'Alert Count', 'Non-Alert Count', 'select_probablity',  'a1c1', 'a0c1']]



import numpy as np
import pandas as pd

# Function to run the simulation based on FP (False Positive) logic with group selection
def run_simulation_non_compromised(severity_decay_factors, num_events=100, max_fp=1):
    # Define the rule data
    rules_data = {
        'Rule ID': [
            2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509,
            5300, 5301, 5302, 5303, 5304, 5305, 5306,
            5901, 5902, 5903, 5904, 5905,
            5400, 5401, 5402, 5407, 5403, 5404, 5405, 5406,
            9100, 9101, 9102,
            5600, 5601, 5602, 5603, 5604, 5631,
            92020, 92022, 92026, 92030, 92031, 92033, 92034, 92035,
            92037, 92038, 92039, 92040, 92046, 92055, 92056, 92062,
            92080, 92023, 92024
        ],
        'Severity Level': [
            6, 11, 6, 10, 4, 4, 4, 4, 6,
            1, 6, 10, 4, 4, 5, 1,
            9, 9, 4, 9, 1,
            1, 6, 4, 4, 5, 11, 6, 6,
            1, 1, 1,
            1, 6, 4, 6, 6, 11,
            7, 4, 15, 4, 4, 4, 4, 4, 4, 13, 4, 13, 13, 13, 15, 15, 8, 9, 15
        ]
    }

    # Define rule groups (each group as a list)
    rule_groups = [
        [2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509],
        [5300, 5301, 5302, 5303, 5304, 5305, 5306],
        [5901, 5902, 5903, 5904, 5905],
        [5400, 5401, 5402, 5407, 5403, 5404, 5405, 5406],
        [9100, 9101, 9102],
        [5600, 5601, 5602, 5603, 5604, 5631],
        [92020, 92022, 92026, 92030, 92031, 92033, 92034, 92035, 92037], 
        [92038, 92039, 92040, 92046, 92055, 92056, 92062, 92080, 92023, 92024]
    ]

    # Create a mapping of each rule to its group
    rule_to_group = {}
    for group in rule_groups:
        for rule in group:
            rule_to_group[rule] = group

    # Create DataFrame
    df_rules = pd.DataFrame(rules_data)

    # Function to map severity level to its class
    def get_severity_class(severity_level):
        if 1 <= severity_level <= 5:
            return 1
        elif 6 <= severity_level <= 10:
            return 2
        elif 11 <= severity_level <= 15:
            return 3
        else:
            return None

    # Assign the user-selected decay factors to the corresponding rules
    df_rules['Severity Decay Factor'] = df_rules['Severity Level'].apply(
        lambda severity: severity_decay_factors[get_severity_class(severity)] if get_severity_class(severity) else 0
    )

    # Define a function to calculate FP using the formula
    def calculate_fp(severity_level, severity_decay_factor, max_fp):
        return max_fp * np.exp(-severity_decay_factor * severity_level)

    # Apply the FP formula to each rule in the DataFrame
    df_rules['FP'] = df_rules.apply(lambda x: calculate_fp(x['Severity Level'], x['Severity Decay Factor'], max_fp), axis=1)

    # Normalize probabilities (as in previous code)
    A = 1  # scaling factor
    B = 1.1  # logistic growth rate (as given)
    C = 6  # midpoint of the logistic function (as given)

    # Calculate logistic frequencies using the formula: A / (1 + exp(B * (severity_level - C)))
    df_rules['Probability'] = A / (1 + np.exp(B * (df_rules['Severity Level'] + 1 - C)))
    df_rules['Normalized Probability'] = df_rules['Probability'] / df_rules['Probability'].sum()

    # Initialize counters for rule selection
    df_rules['Alert Count'] = 0
    df_rules['Non-Alert Count'] = 0
    df_rules['False Positive Count'] = 0
    df_rules['Total Selected'] = 0

    # Simulate the rule selection process
    for _ in range(num_events):
        # Randomly choose one rule based on probability
        selected_rule = np.random.choice(df_rules['Rule ID'], p=df_rules['Normalized Probability'])

        # Get the group of the selected rule
        group = rule_to_group[selected_rule]

        # Select all rules in the group
        for rule_id in group:
            fp_value = df_rules.loc[df_rules['Rule ID'] == rule_id, 'FP'].values[0]
            df_rules.loc[df_rules['Rule ID'] == rule_id, 'Total Selected'] += 1

            # Check if the rule triggers an alert or not based on FP
            if np.random.rand() < fp_value:  # False positive check (alert)
                df_rules.loc[df_rules['Rule ID'] == rule_id, 'False Positive Count'] += 1
                df_rules.loc[df_rules['Rule ID'] == rule_id, 'Alert Count'] += 1
            else:
                df_rules.loc[df_rules['Rule ID'] == rule_id, 'Non-Alert Count'] += 1

    # Calculate a1c0 (alert/num_events) and a0c0 (non-alert/num_events)
    df_rules['a1c0'] = df_rules['Alert Count'] / (df_rules['Alert Count']+ df_rules['Non-Alert Count'])
    df_rules['a0c0'] = df_rules['Non-Alert Count'] / (df_rules['Alert Count']+ df_rules['Non-Alert Count'])
    df_rules['select_probablity'] = df_rules['Total Selected'] / float(num_events)

    # Calculate false positive ratio (FP/Total Selected)
    df_rules['fp_ratio'] = df_rules['False Positive Count'] / df_rules['Total Selected']
    df_rules['fp_ratio'] = df_rules['fp_ratio'].fillna(0)  # Handle NaN if Total Selected is 0

    # Return the DataFrame with the selected columns for display
    return df_rules[['Rule ID', 'Severity Level', 'Total Selected', 'Alert Count', 'Non-Alert Count', 'select_probablity', 'a1c0', 'a0c0']]

def rule_select(select_probablity_map,prabablity_list):
    aa=np.random.uniform(0, 1)
    for i in range(0, len(select_probablity_map)):
        if aa<=select_probablity_map[i]:
            return prabablity_list[i]

##########################################################################################################################
# Example usage of the compromised simulation function:
    # 1: (0.010, 0.021),  # Class 1 (Severity levels 1-5)
    # 2: (0.019, 0.035),  # Class 2 (Severity levels 6-10)
    # 3: (0.033, 0.046)   # Class 3 (Severity levels 11-15)
# severity_growth_factors = {1: 0.021, 2: 0.025, 3: 0.040}
# num_events = 10000
# run_simulation_compromised(severity_growth_factors, num_events, min_fn_level = 0)

##########################################################################################################################

# Example usage of the non_compromised simulation function:
# Class 1: 0.183 ~ 0.375  
# Class 2: 0.157 ~ 0.160
# Class 3: 0.151 ~ 0.307

# severity_decay_factors = {1: 0.375, 2: 0.160, 3: 0.307}
# num_events = 10000
# run_simulation_non_compromised(severity_decay_factors, num_events, max_fp=1)
##########################################################################################################################

severity_growth_factors = {1: 0.003, 2: 0.005, 3: 0.008}
num_events = 10000
c1=run_simulation_compromised(severity_growth_factors, num_events, min_fn_level = 0)
severity_decay_factors = {1: 0.04, 2: 0.03, 3: 0.03}
num_events = 10000
c2=run_simulation_non_compromised(severity_decay_factors, num_events, max_fp=1)
print(c1)
print(c2)



select_probablity_list=[]
prabablity_list=[]

for group_1111 in rule_groups:
    prabablity_list_temp=[]
    select_probablity_temp=0

    for i in range(0, len(c1)):
        
        if c1.iloc[i]['Rule ID'] not in group_1111:
            continue

        select_probablity=(c1.iloc[i]['select_probablity']+c2.iloc[i]['select_probablity'])/2.0
        a1c1=c1.iloc[i]['a1c1']
        a0c1=c1.iloc[i]['a0c1']
        a1c0=c2.iloc[i]['a1c0']
        a0c0=c2.iloc[i]['a0c0']
        a1c1=np.clip(a1c1, 0.01, 0.99)
        a0c1=np.clip(a0c1, 0.01, 0.99)
        a1c0=np.clip(a1c0, 0.01, 0.99)
        a0c0=np.clip(a0c0, 0.01, 0.99)
        select_probablity_temp=select_probablity
        prabablity_list_temp.append([a1c1,a1c0,a0c1,a0c0])

    select_probablity_list.append(select_probablity_temp)    
    prabablity_list.append(prabablity_list_temp)

select_probablity_map=[]
for i in range(0, len(select_probablity_list)):
    select_probablity_map.append(sum(select_probablity_list[0:i+1]))

if __name__ == "__main__":
    for i in range(3):
        print(rule_select(select_probablity_map,prabablity_list))