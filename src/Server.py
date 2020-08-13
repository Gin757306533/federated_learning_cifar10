import tensorflow as tf
from tqdm import tqdm

from Client import Clients
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def buildClients(num, local_client_number=1):
    learning_rate = 0.0001
    num_input = 32  # image shape: 32*32
    num_input_channel = 3  # image channel: 3
    num_classes = 10  # Cifar-10 total classes (0-9 digits)

    #create Client and model
    return Clients(input_shape=[None, num_input, num_input, num_input_channel],
                  num_classes=num_classes,
                  learning_rate=learning_rate,
                  clients_num=num,
                   local_client_number=local_client_number
                   )


def run_global_test(client, global_vars, test_num):
    client.set_global_vars(global_vars)
    acc, loss = client.run_test(test_num)
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(
        ep + 1, test_num, acc, loss))
    return acc, loss


#### SOME TRAINING PARAMS ####
CLIENT_NUMBER = 4
LOCAL_CLIENT_NUMBER = 5
CLIENT_RATIO_PER_ROUND = 0.5
LOCAL_CLIENT_RATIO_PER_ROUND = 0.80
epoch = 10


#### CREATE CLIENT AND LOAD DATASET ####
client = buildClients(CLIENT_NUMBER, LOCAL_CLIENT_NUMBER)
import json
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)



#### BEGIN TRAINING ####
global_vars3 = client.get_client_vars()
for client_number in [5, 8]:
    for lower_client_number in [2, 3]:
        print(client_number, lower_client_number)
        CLIENT_NUMBER = client_number
        LOCAL_CLIENT_NUMBER = lower_client_number
        key = str(client_number) + "_" + str(lower_client_number)
        global_vars = client.get_client_vars()
        client.set_global_vars(global_vars3)
        for ep in range(epoch):
            random_clients = client.choose_clients(CLIENT_RATIO_PER_ROUND)
            global_weights2 = client.get_client_vars()
            init_local_weights = [global_weights2 for _ in range(CLIENT_NUMBER)]
            for client_id in random_clients:
                local_random_clients = client.choose_local_clients(LOCAL_CLIENT_RATIO_PER_ROUND)
                local_client_vars_list = [[] for _ in range(LOCAL_CLIENT_NUMBER)]
                client_local_vars_sum = None
                for local_client_id in local_random_clients:
                    client.set_global_vars(init_local_weights[client_id])
                    client.train_epoch(cid=client_id, lcid=local_client_id)
                    current_client_vars = client.get_client_vars()
                    # sum it up
                    if client_local_vars_sum is None:
                        client_local_vars_sum = current_client_vars
                    else:
                        for cv, ccv in zip(client_local_vars_sum, current_client_vars):
                            cv += ccv
                # average_weights =  client_local_vars_sum / local_random_clients
                average_weights = []
                for var in client_local_vars_sum:
                    average_weights.append(var / len(local_random_clients))
                init_local_weights[client_id] = average_weights



            # We are going to sum up active clients' vars at each epoch
            client_vars_sum = None

            # Train with these clients
            for client_id in random_clients:
                # Restore global vars to client's model
                client.set_global_vars(init_local_weights[client_id])

                # train one client
                client.train_epoch(cid=client_id, lcid=0)

                # obtain current client's vars
                current_client_vars = client.get_client_vars()

                # sum it up
                if client_vars_sum is None:
                    client_vars_sum = current_client_vars
                else:
                    for cv, ccv in zip(client_vars_sum, current_client_vars):
                        cv += ccv

            # obtain the avg vars as global vars
            global_vars = []
            for var in client_vars_sum:
                global_vars.append(var / len(random_clients))

            # run test on 1000 instances
            acc, loss = run_global_test(client, global_vars, test_num=600)
            with open("acc_loss.txt", "r") as f:
                import json
                dic = json.load(f)

            with open("acc_loss.txt", "w") as f:
                import json
                small_dic = dic.get(key, {})
                small_dic[ep] = [acc, loss]
                dic[key] = small_dic
                json.dump(dic, f, cls=MyEncoder)

#### FINAL TEST ####
run_global_test(client, global_vars, test_num=10000)