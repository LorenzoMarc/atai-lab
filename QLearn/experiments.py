import matplotlib.pyplot as plt
import matplotlib
import csv
import pandas as pd
import random

df = pd.read_csv("NewOutput.csv", index_col=False)
#df= df.drop(columns='None')

print(df.tail(5))



learning_rate = df['Learning Rate']
# print ("Learning Rate: ", learning_rate)

lr_avg = df.groupby(['Learning Rate']).mean()
# print ("lr_avg: ", lr_avg)

score = df['Score']
#print ("score: ", score)

avg_score = lr_avg["Score"]
#print ("avg_score: ", avg_score)

my_lr = learning_rate.unique()

plt.hist2d(my_lr, avg_score)



#plt.show()

fig = plt.figure()
ax = plt.subplot(111)
ax.bar(my_lr, avg_score, width=0.1) # , width=1, color='r'

plt.xlabel("Learning rate")
plt.ylabel("Score")
plt.title("Score results based on learning rate values")

plt.show()

#Where run == 0 and Success= True
'''
run_zero= df.loc[df['Run'] == 0]

success = run_zero.loc[df['Success']== True]

print ("successfull episodies: \n", success)



success_run_zero= df[(df['Run'] == 5) & (df['Success'] == True)]
print("type of 5: ", type(5))
print ("Success:\n", success_run_zero)

execution_times = success_run_zero['ExecutionTime']
print ("execution time: ", execution_times)

episode_numb = success_run_zero['Episode']
print ("episode number: ", episode_numb)

plt.plot(episode_numb,execution_times )

#plt.show()

#Ora facciamo il tempo di esecuzione medio per ogni run e vediamo
# quali sono i tempi di esecuzione medi minori (di quanto)
# e vediamo quali sono i valori di lr e gamma corrispondenti


avg = success_run_zero.groupby(['Run']).mean()

print ("avg type: ", type(avg))
#print(success_run_zero)

print ("time execution avg of run = 5: \n\n", avg["ExecutionTime"])

print ("AVG: \n", avg)
print ("avg column type: ", type(avg["ExecutionTime"]))


#Ci sono 24 run totali
runs = []
x = 0
while (x <= 24):
    runs.append(x)
    x += 1
execTime =[]
for i in runs:
    #print ("i: ", i, "type: ", type(i))
    successfull_run = df[(df['Run'] == i) & (df['Success'] == True)]

    #print("\nSUCCESS RUN {}: VALUE\n{}".format(i, successfull_run))
    avg = successfull_run.groupby(['Run']).mean()

    exec_time = avg["ExecutionTime"].to_numpy()
    print ("exec_time: ", exec_time, " exec time value: ", exec_time[0])
    execTime.append(exec_time[0])
    print ("array execTime: ", execTime)



    #type(avg["ExecutionTime"]) = series
    print ("Run: ", i,  " has avg execution time: ", avg["ExecutionTime"].to_numpy(),
            " with lr: ", avg["LearningRate"].to_numpy(), " and gamma: ", avg["Gamma"].to_numpy())


#### MAKING THE PLOT #######
execution_times = success_run_zero['ExecutionTime']


episode_numb = success_run_zero['Episode']
print ("episode number: ", episode_numb)

plt.plot(episode_numb,execution_times )

#plt.show()
'''

