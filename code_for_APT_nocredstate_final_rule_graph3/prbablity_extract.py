import csv

def probability_extract():
    probability_pa_pc=[]
    with open('probability.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i,row in enumerate(spamreader):
            #print(row)
            if i>0:
                probability_pa_pc.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        print(probability_pa_pc)
    return probability_pa_pc
