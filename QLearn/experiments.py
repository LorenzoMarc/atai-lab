import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("NewOutput.csv", index_col=False)

print(df.tail(5))



lr_avg = df.groupby(['Learning Rate']).mean()

avg_score = lr_avg["Score"]

learning_rate = df['Learning Rate'].unique()

plt.hist2d(learning_rate, avg_score)



#plt.show()

fig = plt.figure()
ax = plt.subplot(111)
ax.bar(learning_rate, avg_score, width=0.1) # , width=1, color='r'

plt.xlabel("Learning rate")
plt.ylabel("Score")
plt.title("Score results based on learning rate values")

plt.show()


