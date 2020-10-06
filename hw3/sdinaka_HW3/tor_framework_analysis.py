import csv
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles 

with open('tor.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

diction={}
headers=data[0]
print(headers)

for row in data[1:]:
    for index in range(len(row)):
        if headers[index] not in diction.keys():
            set1=list()
        else:
            set1=diction.get(headers[index])
            set1.append(row[index])
        
        diction[headers[index]]= set1

def first_five():  
    country_code=diction['Country Code']
    code_count={}
    for code in country_code:
        if code in code_count:
            code_count[code]=code_count.get(code)+1
        else:
            code_count[code]=1;
    newA = dict(sorted(code_count.items(), key=itemgetter(1), reverse=True)[:5])
    print(newA)

def first_five_bw():  
    relays=diction['Router Name']
    bandw=diction['ConsensusBandwidth']
    bw_measure={}
    for i in range(len(relays)):
        if(bandw[i]!='None'):
            bw_measure[relays[i]]=int(bandw[i])
    newA = dict(sorted(bw_measure.items(), key=itemgetter(1), reverse=True)[:5])
    print(newA)


def ven():
    exit_r=diction['Flag - Exit']
    exit_no=0
    exit_no_bw=0
    entry_no=0
    entry_no_bw=0
    ex_er=0
    ex_er_bw=0
    mid_no=0
    mid_no_bw=0
    entry_r=diction['Flag - Guard']
    print(entry_r)
    bandw=diction['ConsensusBandwidth']
    bw_measure={}
    for i in range(len(exit_r)):
        flag=0
        if exit_r[i]=='1' and entry_r[i]=='1':
            ex_er+=1;
            flag=1
            if bandw[i]!='None':
                ex_er_bw+=int(bandw[i])
                
        if entry_r[i]=='1' and flag==0:
            entry_no+=1
            flag=1
            if bandw[i]!='None':
                entry_no_bw+=int(bandw[i])

        if exit_r[i]=='1' and flag==0:
            exit_no+=1
            flag=1
            if bandw[i]!='None':
                exit_no_bw+=int(bandw[i])
        
        if flag==0:
            mid_no+=1;
            if bandw[i]!='None':
                mid_no_bw+=int(bandw[i])
        
        
    v3 = venn3(subsets = {'100':50, '010':50,'001':50, '110':30},set_labels = ('', '', ''))
    v3.get_patch_by_id('100').set_color('red')
    v3.get_patch_by_id('010').set_color('yellow')
    v3.get_patch_by_id('001').set_color('blue')
    v3.get_patch_by_id('110').set_color('orange')

    v3.get_label_by_id('100').set_text('entry_node\n'+str(entry_no)+" \n"+str(entry_no_bw))
    v3.get_label_by_id('010').set_text('exit_node\n'+str(exit_no)+" \n"+str(exit_no_bw))
    v3.get_label_by_id('001').set_text('middle node\n'+str(mid_no)+" \n"+str(mid_no_bw))
    v3.get_label_by_id('110').set_text('entry and exit\n'+str(ex_er)+" \n"+str(ex_er_bw))
    plt.show()


first_five()

first_five_bw()

ven()
