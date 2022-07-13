#!/usr/bin/env python
# encoding: utf-8

__author__ = 'luodejin'

'''
@author: luodejin
@license: (C) Copyright 2019-2029, Beijing TaoYou World Technology Corporation Limited.
@contact: luodejin2011@163.com
@software: maimai
@file: pickle_utils
@time: 2022/7/13 4:03 下午
@desc:
'''
import torch
# import pickle

# # 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
# f = open('datasets/cvrp_50_10000.pkl','rb')
# data = pickle.load(f)
# print(data[0])

# index1 = torch.tensor([[0],[2],[1]])
# index2 = torch.tensor([[3],[3],[4]])
#
# dim0, dim1 = index1.shape
#
# data = None
# cnt = 0
# for i in range(dim0):
#     for j in range(dim1):
#         inner_data = torch.range(index1[i][j], index2[i][j])
#         print(inner_data)
#         inner_data = inner_data.expand(1, inner_data.size(0))
#         print(inner_data)
#         if cnt == 0:
#             data = inner_data
#         else:
#             data = torch.cat((data, inner_data), 0)
#         cnt += 1
# print(data)


# def crossover(solution1_, solution2_, index1, index2, is_perturb=False):
#     batch_size = solution1_.size(0)
#
#     solution1, solution2 = None, None
#     for i in range(batch_size):
#         new_c1, new_c2 = single_crossover(solution1_[i], solution2_[i], index1[i], index2[i])
#         if i == 0:
#             solution1, solution2 = new_c1, new_c2
#         else:
#             solution1 = torch.cat((solution1, new_c1), 0)
#             solution2 = torch.cat((solution2, new_c2), 0)
#
#     return solution1, solution2
#
#
# def single_crossover(f1, f2, cro1_index, cro2_index):
#     new_c1_f, new_c1_m, new_c1_b = torch.empty(0), f1[cro1_index:cro2_index + 1], torch.empty(0)
#     new_c2_f, new_c2_m, new_c2_b = torch.empty(0), f2[cro1_index:cro2_index + 1], torch.empty(0)
#     cnt1, cnt2 = 0, 0
#     for index in range(4):
#         if cnt1 < cro1_index:
#             if f2[index] not in new_c1_m:
#                 new_c1_f = torch.cat((new_c1_f, f2[index].expand(1)), 0)
#                 cnt1 += 1
#         else:
#             if f2[index] not in new_c1_m:
#                 new_c1_b = torch.cat((new_c1_b, f2[index].expand(1)), 0)
#     for index in range(4):
#         if cnt2 < cro1_index:
#             if f1[index] not in new_c2_m:
#                 new_c2_f = torch.cat((new_c2_f, f1[index].expand(1)), 0)
#                 cnt2 += 2
#         else:
#             if f1[index] not in new_c2_m:
#                 new_c2_b = torch.cat((new_c2_b, f1[index].expand(1)), 0)
#     new_c1 = torch.cat((new_c1_f, new_c1_m, new_c1_b), -1)
#     new_c2 = torch.cat((new_c2_f, new_c2_m, new_c2_b), -1)
#     return new_c1.unsqueeze(0), new_c2.unsqueeze(0)
#
#
# f1 = torch.Tensor([[1, 2, 3, 4], [1, 3, 2, 4]])
# f2 = torch.Tensor([[3, 4, 1, 2], [4, 1, 2, 3]])
# print(crossover(f1, f2, [1, 2], [2, 3]))


