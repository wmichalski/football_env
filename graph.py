import matplotlib.pyplot as plt

f = open('logs.txt', 'r')
logs = f.read()
values = logs.split()
floats = []
for value in values:
    floats.append(float(value))

f.close()

fig = plt.figure()
plt.xlabel('episodes')
plt.ylabel('average score in last 100 episodes')
plt.plot(floats)
plt.show()