#####
# Aleksandra Mazur
#####

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from functools import partial

domain = (0,1)
N = 10
a = lambda x : 2
b = lambda x : -3
c = lambda x : 1
f = lambda x : x**2
beta = 1
gamma = 2
uR = -1

V = []
L = []
B = []
step = (domain[1] - domain[0]) / N

def derivative(f):
    precision = 0.000001
    return lambda x: ( f( x + precision ) - f( x - precision ) ) / ( 2 * precision )

def functionMultiply(f1, f2):
    return lambda x: f1(x)*f2(x)
    
# Pyramids
for i in range(0, N + 1):
    V.append( partial( lambda i, x: (
        ( (1 / (step * i - step * (i - 1))) * x - step * (i - 1) / (step * i - step * (i - 1))) if (step * i > x > step * (i - 1)) 
        else (((step * (i + 1) / (step * (i + 1) - step * i)) - x * (1 / (step * (i + 1) - step * i))) if (step * i <= x < step * (i + 1)) 
        else 0.0)), i))

# Shift
shift =  lambda x: uR * V[N](x)

# Matrix L
for i in range(0, N):
    L.append( quad ( functionMultiply ( V[i], f ), domain[0], domain[1] )[0] - gamma * V[i](0) )
L.append(uR)

# Matrix B
for v in range(0, N):
    row = []
    for u in range(0, N + 1):
        row.append(
            (-1) * quad( functionMultiply ( functionMultiply ( derivative(V[u]), derivative(V[v])), a), domain[0], domain[1])[0]
            + quad( functionMultiply ( functionMultiply ( derivative(V[u]), V[v]), b), domain[0], domain[1])[0]
            + quad( functionMultiply ( functionMultiply (V[u], V[v]), c), domain[0], domain[1])[0]
            - beta * V[u](0) * V[v](0))
    B.append(row)


row = []
for j in range(0, N):
    row.append(0)
row.append(1)
B.append(row)

B = np.array(B)
L = np.array(L)

u = np.linalg.solve(B, L)

linearCombination = []
for i in range(0, N + 1):
    linearCombination.append ( functionMultiply (V[i], lambda x: u[i] ))

# Showing result
X = []
Y = []
for j in np.arange( domain[0], domain[1], 0.001 ):
    X.append( round(j, 4) )
    value = 0
    for i in range(0, N + 1):
        value += linearCombination[i](j)
    Y.append( round(value,4) )

minY = min(Y) - 1
maxY = max(Y) + 1

plt.plot(X,Y)
plt.xlabel('x')
plt.ylabel('y')

plt.title('Output Function')
# plt.ylim(minY, maxY)
plt.show()