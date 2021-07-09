import math

#funcao phi
def phi(x):
    y = 2 - (3/(4*x))
    return y

#definindo k
eplison = 10e-8
k_menor = math.log10(eplison)/math.log10(0.75)
if k_menor//1 == k_menor:
    k = k_menor
else:
    k = int(k_menor) + 1 # como k >= K_menor e k é inteiro, k = K_menor + 1
print('k = {}'.format(k))

a_0 = 2
a_1 = phi(a_0) #constante inicial
i = 1

#loop de iteração 
for c in range(0, k):
    print("a{} = {}".format(i, a_1))

    a_2 = phi(a_1)
    i += 1
    a_1 = a_2
    