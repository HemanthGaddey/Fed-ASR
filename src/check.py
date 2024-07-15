import multiprocessing
from multiprocessing import Pool
import time, random
import concurrent.futures
from src.Client.Client import train_model

def run(id):
    s_t=time.perf_counter()
    t=random.randint(10,20)
    print(f"pid:{id} started")
    if id==0:
        t=17
    elif id==1:
        t=13
    elif id==2:
        t=19
    else:
        t=11
    time.sleep(t)
    e_t=time.perf_counter()

    print("Stats of pid {}: \n delay:{} \n start_time:{} \n end_time:{} \n".format(id,t, s_t, e_t))

def start_train(id):
    # print(f'Client:{id} training is started')
    run(id)
    # print(f'Client:{id} training is completed')

if __name__=="__main__":
    # inputs = [0, 1, 2, 3]

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(run, i) for i in inputs]

    #     # Wait for all processes to finish
    #     concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)


    pool=multiprocessing.Pool(processes=5)
    inputs=[0,1,2,3,4]
    results = pool.map_async(start_train, inputs)
    # results=[pool.apply_async(run, (i,)) for i in inputs]
    process_result=results.get()

    # time.sleep(1)
    pool.close()
    pool.join()

    # outputs=outputs_async.get()
    # outputs=pool.imap_unordered(run, inputs)
    # for i in outputs:
    #     print("here",i)    