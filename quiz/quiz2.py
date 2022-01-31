import numpy as np
x = -1.8; y = -0.2 # Insert
z = 20 + x**2 + y**2 -10*np.cos(2*np.pi*x)-10*np.cos(2*np.pi*y)
for _ in range(100):
    x = x + 0.001*(2*x + 20*np.pi*np.sin(2*np.pi*x))
    y= y + 0.001*(2*y + 20*np.pi*np.sin(2*np.pi*y))
print(x, y)
