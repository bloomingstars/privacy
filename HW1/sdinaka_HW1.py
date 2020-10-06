import re, sys
import math, random
import numpy as np

#### BEGIN----- functions to read movie files and create db ----- ####

def add_ratings(db, chunks, num):
    if(chunks[0] not in db):
        db[chunks[0]] = {}
    db[chunks[0]][num] = int(chunks[2])

def read_files(db, num):
    movie_file = "G:\Privacy\HW1\hw1-files\movies/"+num
    ratings = []
    fo = open(movie_file, "r")
    r = 0
    for line in fo:
        chunks = re.split(",", line)
        chunks[len(chunks)-1] = chunks[len(chunks)-1].strip()
        add_ratings(db, chunks, num)

#### END----- functions to read movie files and create db ----- ####

def score(w, p, aux, r):
    '''
    Inputs: weights of movies, max rating per moive, auxiliary information, and a record, 
    Returns the corresponding score
    '''
    
    aux_diff={}
    supp=calc_supp_aux(aux);
    for key,aux_val in aux.items():
        if key!=None:
            if r.get(key)!=None and r.get(key)>=0:
                T=abs(aux_val-r.get(key))
                aux_diff[key]=w.get(key)*(1 - T/p.get(key))
    return sum(aux_diff.values())/supp
    pass



def compute_weights(db):
    '''
    Input: database of users
    Returns weights of all movies
    '''
    #### ----- your code here ----- ####
    table=["03124", "06315", "07242", "16944", "17113", "10935", "11977", "03276", "14199", "08191", "06004", "01292", "15267", "03768", "02137"]
    hash_table={}
    for item in table:#initialise
        hash_table[item]=0
    for user,di in db.items():#calculate supp
        if(di is not None):
            for movie,rating in di.items():
                if rating!=None and rating>=0 and movie in hash_table:
                    hash_table[movie]=hash_table.get(movie)+1
                    
    for key,item in hash_table.items():#take log(supp)
        hash_table[key]=1/math.log(item,2)
    display_table(hash_table)
    return hash_table
    pass



#### BEGIN----- additional functions ----- ####
def display_table(hash_table):
    print("movie | weight")
    for item in hash_table:
        print(item+" : "+str(hash_table.get(item)))
    pass

def calc_p(db,aux):
    maxi={}
    mini={}
    p={}
    table=["03124", "06315", "07242", "16944", "17113", "10935", "11977", "03276", "14199", "08191", "06004", "01292", "15267", "03768", "02137"]
    for item,aux_val in aux.items():
        maxi[item]=aux_val
        mini[item]=aux_val
    for user,di in db.items():
        if(di is not None):
            for movie,rating in di.items():
                if movie in aux.keys() and rating!=None and rating>=0:
                    if(maxi.get(movie)<rating):
                        maxi[movie]=rating
                    if(mini.get(movie)>rating):
                        mini[movie]=rating
    for movie in aux:
        p[movie]=maxi.get(movie)-mini.get(movie)       
    return p
    pass

def all_scores(db,aux):
    p=calc_p(db,aux)
    #print (p)
    score_user={}
    for user,di in db.items():
        if(di is not None):
            score_user[user]=score(w, p, aux, di)
    return score_user
    
def calc_max(db,aux):
    all_score=all_scores(db,aux)
    max_score= max(all_score.values())
    return max_score

def calc_max2(db,aux,maxm):
    max2=list(all_scores(db,aux).values())
    max2.remove(maxm)
    max2_val=max(max2)
    return max2_val
    
def find_user(db,aux):
    all_score=all_scores(db,aux)
    max_val=calc_max(db,aux)
    user_id=-1
    for key,value in all_score.items():
        if(value==max_val):
            user_id=key
    return user_id

def compare_rating(user_id_db,db,aux):
    r=db.get(user_id_db)
    diff_table={}
    print("movie | user_rating | aux_rating | difference")
    for item in aux:
        diff_table[item]=abs(r.get(item)-aux.get(item))
        print(item+" : \t"+str(r.get(item))+"\t "+str(aux.get(item))+"\t\t"+str(diff_table.get(item)))
    return diff_table

def calc_supp_aux(aux):
    supp=0
    for key,rating in aux.items():
        if key!=None and rating!=None and rating>=0:
            supp+=1
    return supp

def calc_metric(w):
    gamma=0.1
    sum_w=0
    supp=calc_supp_aux(aux);
    for key,aux_val in aux.items():
        if key!=None and aux_val!=None and aux_val>=0:
                sum_w+=w.get(key)
    return gamma*sum_w/supp
    return metric
    
def compare_w_metric(diff,metric):
    if metric<=diff:
        return "user identified is right: metric lesser than threshold"
    else:
        return "threshold greater than difference between max and second max value"        

#### END----- additional functions ----- ####

if __name__ == "__main__":
    db = {}
    files = ["03124", "06315", "07242", "16944", "17113",
            "10935", "11977", "03276", "14199", "08191",
            "06004", "01292", "15267", "03768", "02137"]
    for file in files:
        read_files(db, file)

    aux = { '14199': 4.5, '17113': 4.2, '06315': 4.0, '01292': 3.3,
            '11977': 4.2, '15267': 4.2, '08191': 3.8, '16944': 4.2,
            '07242': 3.9, '06004': 3.9, '03768': 3.5, '03124': 3.5}
    
    #### ----- your code here ----- ####
    print("1a");
    w=compute_weights(db)
    #1b
    print("1b");
    n_users_in_db=0
    for user,di in db.items():#number of users
        n_users_in_db+=1
    print (n_users_in_db);
    maxm=calc_max(db, aux)
    print (maxm)
    max2=calc_max2(db,aux,maxm)
    print (max2)
    #1c
    print("1c");
    user_id_db=find_user(db,aux)
    print (user_id_db)
    compare_rating(user_id_db,db,aux)
    #1d
    print("1d");
    metric=calc_metric(w)
    print(metric)
    diff=maxm-max2
    print(diff)
    print(compare_w_metric(diff,metric))
    
