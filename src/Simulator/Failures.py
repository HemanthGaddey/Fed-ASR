import numpy as np

def client_fail(num_clients=10, num_rounds=50, distribution='random', failure_rate=0.03,permanent=True):
    l = []
    if(distribution=='gaussian'):
        mu, sigma = 0, 1 # mean and standard deviation
        #s = np.random.normal(mu, sigma, 1000)
        for i in range(num_clients):
            l.append(-1)
            for i in range(1,num_rounds):
                if(np.abs(np.random.normal(mu, sigma, 1)[0]) > (1-failure_rate)):
                    l[-1]=i
                    break
    else:
        for i in range(num_clients):
            l.append(-1)
            for i in range(1,num_rounds):
                if(np.random.random() < failure_rate):
                    l[-1]=i
                    break
    
    x=0
    for i in l:
        if(i!=-1):
            x+=1
            
    return l,x