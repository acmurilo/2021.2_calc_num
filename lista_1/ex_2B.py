import matplotlib.pyplot as plt
import numpy as np

h=1/20; a=0; b=1; n=20; p=0

x = [] #valores de q para o gráfico
y = [] #valores de Diag(q) para o gráfico

def Ah(x):
    if x == 1:
        return (2+h**2*q)
    else:
        if x==2:
            y = 2+h**2*q-((-1-h/2*p)*(-1+h/2*p)/(2+h**2*q))
        else:
            y = 2+h**2*q-((-1-h/2*p)*(-1+h/2*p)/(Ah(x-1)))
        return y
        
for q in range (-10,1):
    Diag = Ah(1)
    for i in range(2,n+1):
        Diag = Diag*Ah(i)

    x.append(q)
    y.append(Diag)


plt.plot(x, y, '-o')

plt.xlabel('Valores de q')
plt.ylabel('Valores de Diag(q)')
plt.title('Gráfico de Valores da função minDiag(q) por q')
plt.xticks(np.arange(-10, 1, step=1))
plt.grid()
plt.show()
