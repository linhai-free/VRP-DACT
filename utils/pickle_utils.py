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


# def two_opt(solution, first, second, is_perturb=False):
#     rec = solution.clone()
#
#     # fix connection for first node
#     argsort = solution.argsort()
#     pre_first = argsort.gather(1, first)
#     pre_first = torch.where(pre_first != second, pre_first, first)
#     rec.scatter_(1, pre_first, second)
#
#     # fix connection for second node
#     post_second = solution.gather(1, second)
#     post_second = torch.where(post_second != first, post_second, second)
#     rec.scatter_(1, first, post_second)
#
#     # reverse loop:
#     cur = first
#     for i in range(5):
#         cur_next = solution.gather(1, cur)
#         rec.scatter_(1, cur_next, torch.where(cur != second, cur, rec.gather(1, cur_next)))
#         cur = torch.where(cur != second, cur_next, cur)
#
#     return rec
#
#
# solution = torch.Tensor([[4,2,0,1,3]]).long()
# first = torch.Tensor([[4]]).long()
# second = torch.Tensor([[2]]).long()
# print(two_opt(solution, first, second))


# def insert_by_index(solution, index1, index2, is_perturb=False, need_protect=False, protect_start_index=None):
#     """insert element of index1 to index2
#     :param solution:
#     :param index1
#     :param index2
#     :param is_perturb
#     :param need_protect
#     :param protect_start_index
#     :return:
#     """
#     rec = solution.clone()
#     first = rec.gather(1, index1)
#     shape = index1.size()
#
#     # shift left
#     for i in range(4):
#         index_i = torch.full(shape, i)
#         index_i1 = torch.full(shape, i + 1)
#         need_shift_left = torch.logical_and(index1 <= index_i, index_i < index2)
#         if need_protect:
#             need_shift_left = torch.logical_and(need_shift_left, i < protect_start_index)
#             index_i1 = torch.where(i + 1 < protect_start_index, index_i1, index2)
#         rec.scatter_(1, index_i, torch.where(need_shift_left, rec.gather(1, index_i1), rec.gather(1, index_i)))
#
#     # shift right
#     for i in range(5 - 1, 0, -1):
#         index_i = torch.full(shape, i)
#         index_i1 = torch.full(shape, i - 1)
#         need_shift_right = torch.logical_and(index1 >= index_i, index_i > index2)
#         rec.scatter_(1, index_i, torch.where(need_shift_right, rec.gather(1, index_i1), rec.gather(1, index_i)))
#
#     # insert
#     rec.scatter_(1, index2, first)
#
#     return rec
#
#
# def crossover(solution1_, solution2_, index1, index2, is_perturb=False):
#     """
#     :param solution1_: [batch_size, size]
#     :param solution2_: [batch_size, size]
#     :param index1: [batch_size, 1]
#     :param index2: [batch_size, 1]
#     :param is_perturb:
#     :return:
#     """
#     rec1 = solution1_.clone()
#     rec2 = solution2_.clone()
#     argsort1 = rec1.argsort()
#     argsort2 = rec2.argsort()
#
#     shape = index1.size()
#
#     for i in range(5):
#         index_i = torch.full(shape, i).long()
#         con = torch.logical_and(index1 <= i, i <= index2)
#
#         index1_ = torch.where(con, argsort1.gather(1, solution2_.gather(1, index_i)), index_i)
#         index2_ = torch.where(con, argsort2.gather(1, solution1_.gather(1, index_i)), index_i)
#
#         rec1 = insert_by_index(rec1, index1_, index_i, False, True, index1)
#         rec2 = insert_by_index(rec2, index2_, index_i, False, True, index1)
#
#         argsort1 = rec1.argsort()
#         argsort2 = rec2.argsort()
#
#     return rec1, rec2


# solution = torch.Tensor([[1, 3, 0, 2, 4]]).long()
# index1 = torch.Tensor([[4]]).long()
# index2 = torch.Tensor([[0]]).long()
# print(insert_by_index(solution, index1, index2))


# solution1 = torch.Tensor([[1, 2, 3, 4, 0], [1, 3, 0, 2, 4]]).long()
# solution2 = torch.Tensor([[3, 0, 4, 1, 2], [4, 1, 2, 0, 3]]).long()
# index1 = torch.Tensor([[1], [0]]).long()
# index2 = torch.Tensor([[3], [2]]).long()
# print(crossover(solution1, solution2, index1, index2))
