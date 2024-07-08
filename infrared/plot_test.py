import matplotlib.pyplot as plt

a = [1, 2, 3]
b = [5, 12, 4, 62, 5, 2, 3, 1, 1, 1, 1]
c = [2, 3, 41, 2]
d = [8, 3, 5, 2, 4, 8, 1]

plt.subplot(211)
plt.plot(a)
plt.title('a')
plt.subplot(212)
plt.plot(b)
plt.title('b')
plt.tight_layout()

plt.subplot(211)
plt.plot(c)
plt.title('c')
plt.subplot(212)
plt.plot(d)
plt.title('d')
plt.tight_layout()
plt.show()
