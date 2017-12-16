from sklearn.datasets import fetch_rcv1
import math
import random
def read_data(dataset_type):

    rcv1 = fetch_rcv1()
    return rcv1.data, rcv1.target
def get_random_eles_from_list(list_to_select, num_ele):
    return random.sample(list_to_select, num_ele)

def get_batches(data, labels, batch_size):
    data_length = data.shape[0]
    batch_num = (int)(math.ceil(data_length/batch_size))
    batch_list = []
    for i in range(batch_num):
        batch = {}
        if (i+1)*batch_size <= data_length:
            batch['x'] = data[i*batch_size:(i+1)*batch_size]
            batch['y'] = labels[i*batch_size:(i+1)*batch_size]
        else:
            batch['x'] = data[i*batch_size:data_length]
            batch['y'] = labels[i*batch_size:data_length]
        batch_list.append(batch)
    return batch_list
def random_sampling(data, labels, batch_size):
    data_length = data.shape[0]

    indics_list = [i for i in range(data_length)]
    batch_list = []
    for i in range(int(math.ceil(data_length/batch_size))):
        batch = {}
        index_list = get_random_eles_from_list(indics_list, batch_size)
        batch['x'] = data[index_list,:]
        batch['y'] = labels[index_list,:]
        batch_list.append(batch)
    return batch_list
if __name__ == '__main__':
    data, target = read_data('train')
    data = data[:50000]
    target = target[:50000]
    # list1 = [0,3,7,804413]
    # print(data.shape[1])
    # print(data)
    data.todense()
    target.todense()