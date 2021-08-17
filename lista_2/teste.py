from math import factorial as fac
from math import sin, cos, pi, log10
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def r(x):
    return -pi**2*(sin(pi*x) + cos(pi*x))

def binom(a, b):
    return fac(a)/(fac(b)*fac(a - b))

def binom(a, b):
    return fac(a)/(fac(b)*fac(a - b))

def legendre_pol(index, x):
    return sum([binom(index, i)*binom(index+i, i)*((x - 1)/2)**i for i in range(index+1)])

x_given = 0.7688
for i in range(0, 10):
    print(f'P_{i} = {legendre_pol(i, x_given)}')

def integral_trapezoidal(f, a, b, m):
    coeficiente = float(b - a) / m
    somatorio = 0.0
    somatorio += f(a)/2.0
    for i in range(1, m):
        somatorio += f(a + i*coeficiente)
    somatorio += f(b)/2.0
    return somatorio * coeficiente

initial = -1
final = 1

def get_alfa(k, num_divs = 10000):
    return (2*k - 1)/2*integral_trapezoidal(lambda x: r(x)*legendre_pol(k-1, x), initial, final, num_divs)


for i in range(1, 10):
    print(f'α_{i} = {get_alfa(i)}')

for i in range(1, 10):
    print(f'α_{i} = {get_alfa(i)}')

def legendre_duas_integrais(index, x):
    return sum([binom(index, i)*binom(index+i, i)*((x - 1)/2)**(i + 2)*4/((i+1)*(i+2)) for i in range(index+1)])

def F_grande(x, k):
    return sum([get_alfa(i)*legendre_duas_integrais(i-1, x) for i in range(1,k+1)])

def G_k(x, k):
    return F_grande(x, k) + (-1 - F_grande(-1, k))*(1 - x)/2 + (-1 - F_grande(1, k))*(x + 1)/2


for i in [-1, -.7, 0, .3, 1]:
    print(f'G_4({i}) = {G_k(i, 4)}')

for i in [-1, -.7, 0, .3, 1]:
    print(f'G_7({i}) = {G_k(i, 7)}')

k_max_erro = 30
k_min_erro = 2

alfas = {
    'trapezoidal': {
        'm10^4': np.array([get_alfa(i) for i in range(1, k_max_erro+2)]),
        'm10^5': np.array([get_alfa(i, 100000) for i in range(1, k_max_erro+2)])
    }
}

num_subintervalos = 10117
subintervalos_x = np.array([initial + i*(final - initial)/num_subintervalos for i in range(num_subintervalos+1)])

y_of_x_exato = np.sin(np.pi*subintervalos_x) + np.cos(np.pi*subintervalos_x)

def F_grande_otimizado(x, k, alfa_vec):
    ints_legendre = np.array([legendre_duas_integrais(i-1, x) for i in range(1,k+1)])
    return np.sum(alfa_vec[:k]*ints_legendre)

def G_k_otimizado(x, k, alfa_vec):
    return F_grande_otimizado(x, k, alfa_vec) + (-1 - F_grande_otimizado(-1, k, alfa_vec))*(1 - x)/2 + (-1 - F_grande_otimizado(1, k, alfa_vec))*(x + 1)/2

erros_m_k = {
    "trapezio": {
        "m10^4": [],
        "m10^5": [],
    },
    "simpson": {
        "m10^5": []
    }
}

for k in range(k_min_erro, k_max_erro+1):
    G_k = np.array([G_k_otimizado(i, k+1, alfas['trapezoidal']['m10^4']) for i in subintervalos_x])
    erros_m_k['trapezio']['m10^4'].append({'k': k, 'erro': max(np.abs(y_of_x_exato - G_k))})
    print(f'E_{k} = {max(np.abs(y_of_x_exato - G_k))}')

for k in range(k_min_erro, k_max_erro+1):
    G_k = np.array([G_k_otimizado(i, k+1, alfas['trapezoidal']['m10^5']) for i in subintervalos_x])
    erros_m_k['trapezio']['m10^5'].append({'k': k, 'erro': max(np.abs(y_of_x_exato - G_k))})
    print(f'E_{k} = {max(np.abs(y_of_x_exato - G_k))}')

def simpson(f, a, b, n):
    h = float(b - a) / n
    s = 0.0
    s += f(a)/3.0
    for i in range(1, n, 2):
        s += (4/3)*f(a + i*h)
    for i in range(2, n, 2):
        s += (2/3)*f(a + i*h)
    s += f(b)/3.0
    return s * h

def get_alfa_simpson(k, num_divs = 100000):
    return (2*k - 1)/2*simpson(lambda x: r(x)*legendre_pol(k-1, x), initial, final, num_divs)

alfas['simpson'] = {'m10^5': np.array([get_alfa_simpson(i) for i in range(1, k_max_erro+2)])}

for k in range(k_min_erro, k_max_erro+1):
    G_k = np.array([G_k_otimizado(i, k+1, alfas['simpson']['m10^5']) for i in subintervalos_x])
    erros_m_k['simpson']['m10^5'].append({'k': k, 'erro': max(np.abs(y_of_x_exato - G_k))})
    print(f'E_{k} = {max(np.abs(y_of_x_exato - G_k))}')

ks = list(map(lambda x: x.get('k'), erros_m_k['trapezio']['m10^4']))
erros_trap_10_4 = list(map(lambda x: log10(x.get('erro')), erros_m_k['trapezio']['m10^4']))
erros_trap_10_5 = list(map(lambda x: log10(x.get('erro')), erros_m_k['trapezio']['m10^5']))
erros_simp_10_5 = list(map(lambda x: log10(x.get('erro')), erros_m_k['simpson']['m10^5']))
plt.plot(ks,erros_trap_10_4, label='Trapezoidal, m=10^4') 
plt.plot(ks,erros_trap_10_5, label='Trapezoidal, m=10^5') 
plt.plot(ks,erros_simp_10_5, label='Simpson, m=10^5') 
plt.legend()
plt.show()

matriz_coef = np.zeros([3,3])
matriz_indep = np.zeros([3,1])
matriz = np.zeros([3,1])

def matrizF (i,x):
    matriz[0,0] = 1
    matriz[1,0] = x
    matriz[2,0] = x**2
    return matriz([i,0])

for i in range(0,3):
    for j in range (0,3):
        for k in range (2,17):
            matriz_coef[i,j] += matrizF(i, k)*matrizF(j, k)
            matriz_indep[i,0] += matrizF(i, k)*(erros_simp_10_5[k])
Df_Mcoef = pd.DataFrame(matriz_coef)
Df_Mindep = pd.DataFrame(matriz_indep)
print("Matriz dos coeficientes:")
print (Df_Mcoef)
print("\n Matriz dos termos independentes:")
print(Df_Mindep)


