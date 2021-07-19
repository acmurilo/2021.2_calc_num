# Biblioteca utilizada na implementação
import math

# função phi (já com as funções constantes substituídas)
def phi(x):
    return 2 - (3/(4*x))

# Erro aceitável
eplison = 1e-8

# Menor k necessário (Ainda um número real)
k_menor = math.log10(eplison)/math.log10(0.75)

# Como k  inteiro, coloca o menor inteiro maior que k_menor
if k_menor//1 == k_menor:
    k = k_menor
else:
    # k deve ser inteiro e >= K_menor. Logo, k = K_menor + 1
    k = int(k_menor) + 1 
print(f'k = {k}')

# Defininfo a_0 (j substituindo as funções)
a_0 = 2

# Definindo a_1
a_i = phi(a_0)

# Para o loop de iteração ir de 1 a k
i = 1

# Print de todos os coeficientes a_i
for c in range(0, k):
    print(f'a{i} = {a_i}')

    # Tem-se que a_i+1 = phi(a_i)  
    a_i_1 = phi(a_i)
    i += 1
    
    # Redefine a_i para ser printado e utilizado no próximo a_i+1
    a_i = a_i_1
    
