import scipy.io as scio
import numpy as np
def changedataformat(filename):
    
    data = scio.loadmat(filename)
    labels = []
    for i in range(0,len(data['y'])):#expanding 1 dimension to 10
        if data['y'][i][0] == 10:
            labels.append([1,0,0,0,0,0,0,0,0,0])
        elif data['y'][i][0] == 1:
            labels.append([0,1,0,0,0,0,0,0,0,0])
        elif data['y'][i][0] == 2:
            labels.append([0,0,1,0,0,0,0,0,0,0])
        elif data['y'][i][0] == 3:
            labels.append([0,0,0,1,0,0,0,0,0,0])
        elif data['y'][i][0] == 4:
            labels.append([0,0,0,0,1,0,0,0,0,0])
        elif data['y'][i][0] == 5:
            labels.append([0,0,0,0,0,1,0,0,0,0])
        elif data['y'][i][0] == 6:
            labels.append([0,0,0,0,0,0,1,0,0,0])
        elif data['y'][i][0] == 7:
            labels.append([0,0,0,0,0,0,0,1,0,0])
        elif data['y'][i][0] == 8:
            labels.append([0,0,0,0,0,0,0,0,1,0])
        elif data['y'][i][0] == 9:
            labels.append([0,0,0,0,0,0,0,0,0,1])
    return labels,data['X'].transpose().reshape(data['X'].shape[-1],32*32*3)*1.00/255
def convert_test_data(filename):
    data = scio.loadmat(filename)
    return data['X'].transpose().reshape(data['X'].shape[-1],32*32*3)*1.00/255
