import matplotlib.pyplot as plt

f = open('logs.txt', 'r')
logs = f.read()
values = logs.split()
floats = []
for value in values:
    floats.append(float(value))

f.close()

plt.plot(floats)
plt.show()