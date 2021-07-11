h=1/20; a=0; b=1; n=20; p=0

def Ah(x):
    if x == 1:
        return (2+h**2*q)
    else:
        if x==2:
            y = 2+h**2*q-((-1-h/2*p)*(-1+h/2*p)/(2+h**2*q))
        else:
            y = 2+h**2*q-((-1-h/2*p)*(-1+h/2*p)/(Ah(x-1)))
        return y
        
q = -9.9
Diag = Ah(1)
for i in range(2,n+1):
    if Ah(i) < Ah(i-1):
        Diag = Ah(i)

print(Diag)
        