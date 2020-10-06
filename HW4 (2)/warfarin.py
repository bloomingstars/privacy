from sympy import *

def age(a):
    return int(a/10)

def height(inches, feet):
    height=inches*30.48
    height+=feet*2.54
    return height
    
def weight(kg):
    return kg

def race(R):
    if R=='C':
        return 0
    if R=='A':
        return -0.1092
    if R=='B':
        return -0.2760
    if R=='U':
        return -0.1032

def enzyme(Y):
    if(Y=='Y'):
        return 1
    return 0

def amiodarone(Y):
    if(Y=='Y'):
        return 1
    return 0

a0, a1, a2, a3, a4, a5, a6, a7, a8=symbols('a0:%d'%9)

def inititalise():
    A=[0 for i in range(9)]
    return A
def actual_dose():
    return 21

def mode_inversion(expr):
    A=inititalise()
    Result=[];
    mini=0
    minj=0
    min=1000
    B=['AG','AA','GENOTYPE UNKNOWN','*1/*2','*1/*3','*2/*2','*2/*3','*3/*3','GENOTYPE UNKNOWN'];
    for i in range(0,3):
        A[i]=1
        for j in range(3,9):
            A[j]=1
            expr2 = expr.subs(a0, A[0]).subs(a1, A[1]).subs(a2, A[2]).subs(a3, A[3]).subs(a4, A[4]).subs(a5, A[5]).subs(a6, A[6]).subs(a7, A[7]).subs(a8, A[8])

            print('for VKORC1 : ',B[i],' and CYP2C9 : ',B[j],end = '|')

            print('dosage amount : ', expr2,end = '')
            expr2=abs(actual_dose()-expr2)

            print('diff between actual and this dosage amount : ', expr2)

            if(expr2 < 1):
                list1=[i,j,expr2]
                Result.append(list1)

            A[j]=0

        A[i]=0

    for lis in Result:

            print('\nVKORC1 is : ',B[lis[0]] ,'\t CYP2C9 is : ', B[lis[1]] , '\t error : ' , lis[2])


expr= (5.6044 - 0.2614 * age(57) + 0.0087 * height(5,10) + 0.0128 * weight(72) -0.8677*a0 - 1.6974*a1 - 0.4854*a2 - 0.5211*a3 - 0.9357*a4 - 1.0616*a5 - 1.9206*a6 - 2.3312*a7 - 0.2188*a8 + race('C') + 1.1816 *enzyme('Y')- 0.5503 * amiodarone('Y'))**2
mode_inversion(expr)


