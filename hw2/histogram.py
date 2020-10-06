import math, random
import numpy as np
import csv
import array as arr
import matplotlib.pyplot as plt

np.set_printoptions(precision=1)

def read():
    price_array=arr.array('i',[])
    csv.register_dialect('myDialect',   delimiter = '$',skipinitialspace=True)
    with open('G:/Privacy/hw2/IL_employee_salary.csv', 'r') as f:
        reader = csv.reader(f, dialect='myDialect')
        for row in reader:
            if(len(row)==2):
                price_array.append(int(row[1].strip(' \"').replace(',','')))
    return price_array

def plot_hist(a):
    #read from csv
    salary_array=read()
    #creating bins of size 5k in salary_range 50k to 100k
    bin_values = np.arange(start=50000,stop=100000,step=5000)
    plt.xlabel("salary")
    plt.ylabel("no_of_employees")
    #y value increments by 10
    plt.grid(axis='y', alpha=0.75)
    plt.yticks(range(0, len(salary_array), 2))
    plt.xticks(range(50000, 100000, 5000))
    plt.title("salary distribution of employees")
    count=a.hist(label='original',x=salary_array, bins=bin_values, color='#0504aa',alpha=0.9,rwidth=0.9)
    print(count)
    #plt.show()
    return count

def get_bar(a,epsilon):

    sensitivity=1
    scale=sensitivity/epsilon
    print("scale: ", scale)
    bin_values = np.arange(start=50000,stop=100000,step=5000)
    count=plot_hist(a)
    new_array=arr.array('i',[])
    for count_item in count[0]:
        loc,scale = count_item,scale
        noise=np.random.laplace(0,scale,1)
        print(noise," ",count_item," ")
        count_item=count_item+noise
        print('count_item_finals',count_item)
        new_array.append(count_item)
    print(new_array)
    count=a.hist(bin_values[:-1], bin_values, weights=new_array,alpha=0.25,rwidth=0.9 )
    print(count)
    return count

def s_subplots():
    # set width of bar
    barWidth = 0.20

    fig, (a, b)= plt.subplots(1,2, figsize=(40,20))
    # set height of bar
    bars1 = get_bar(a,0.05)[0]
    bars2 = get_bar(a,0.1)[0]
    bars3 = get_bar(a,5.0)[0]
    bars4= plot_hist(a)[0]
 
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    # Make the plot
    b.bar(r1, bars1, color='red', width=0.25, label='epsilon 0.05')
    b.bar(r2, bars2, color='#557f2d', width=0.25,  label='epsilon 0.1')
    b.bar(r3, bars3, color='blue', width=0.25,  label='epsilon 5')
    b.bar(r4, bars4, color='orange', width=0.25,  label='original')

    # Add xticks on the middle of the group bars
    plt.xlabel("salary")
    plt.ylabel("no_of_employees")
    plt.yticks(range(0, 100, 10))
    plt.title("salary distribution of employees")
    plt.xticks([r+barWidth for r in range(len(bars1))],range(50000, 100000, 5000))
 
    # Create legend & Show graphic
    plt.legend()
    plt.show()
    

s_subplots()



