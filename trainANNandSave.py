import numpy as np
import neuralNetwork

# region
# 创建一个神经网络
# hnode = 3
# onode = 3
# lr = 0.5
# n = main.neuralnetwork(inode,hnode,onode,lr)
# # input是一个方括号扩住的list
# input_lists1 = [1.1,1.35,-0.4]
# n.query(input_lists1)
# inode = 3
# data_file = open("mnist_train_100.csv",'r')
# # # 注意区分readline和readlines的区别
# data_list = data_file.readlines()
# # print(len(data_list))
# # print(data_list[0])
# data_file.close()
# #
# # # 除去第一个标记值 后面784个值代表像素 每个值落在0到255之间
# all_values = data_list[1].split(',')
# image_array = np.asarray(all_values[1:], dtype=float).reshape((28,28))
# plt.imshow(image_array,cmap='gray',interpolation='none')
# plt.show()
# endregion

# 28*28
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.2

# create instance

n = neuralNetwork.neuralnetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# load the training file

data_file = open("data/mnist_train.csv",'r')
train_list = data_file.readlines()
data_file.close()

for record in train_list:
    all_values = record.split(',')

    inputs = (np.asarray(all_values[1:],dtype=float)/225.0 *0.99)+0.01
    # initial every output be low
    targets = np.zeros(output_nodes) +0.01
    # target: output from 0-9
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass

n.save_weights()




# load the test

# test_data = open("data/mnist_test.csv",'r')
# test_list = test_data.readlines()
# test_data.close()

# region
# 单个查询和单次的判断后的矩阵查询

# all = test_list[0].split(',')
# print(all[0])
#
# # 查询用网络所得出的权重矩阵
# # 下面实现了得出权重矩阵的最大值 这里涉及到了几个易错的点：1、matrix的flatten化 2. argwhere所返回的是一个多维的数组 3.需要将
# # 查询到的返回数组再变成int形式
# final_matrix = n.query((np.asarray(all[1:],dtype=float)/255.0*0.99)+0.01)
#
# print(final_matrix)
#
# test_value = max(final_matrix)
#
# print(test_value)
# final_matrix = final_matrix.flatten()
# final_index = np.argwhere(final_matrix == test_value)
#
# print(final_index)
#
# index_final = int(final_index[0,0])
# output = int(all[0])
# if index_final == output:
#     print("correct")
# else: print("wrong")

# endregion
# 接下来完成整个test file的查询和记录

# correct = 0.0
# all_times = 0.0
# for data in test_list:
#     all_times+=1.0
#     all = data.split(',')
#     output = int(all[0])
#     final_matrix = n.query((np.asarray(all[1:], dtype=float) / 255.0 * 0.99) + 0.01)
#     test_value = max(final_matrix)
#     final_matrix = final_matrix.flatten()
#     final_index = np.argwhere(final_matrix == test_value)
#     index_final = int(final_index[0, 0])
#
#
#     if output == index_final:
#         correct+=1.0
#     pass
#
# correct_rates = correct/all_times
#
# print(correct_rates)


