from torch.utils.data import Dataset
import torch
import pickle
import os
import numpy as np

class CVRP(object):

    NAME = 'cvrp'  # Capacitiated Vehicle Routing Problem
    
    def __init__(self, p_size, init_val_met = 'greedy', with_assert = False, step_method = '2_opt', P = 250, DUMMY_RATE = 0.2):
        
        self.size = int(np.ceil(p_size * (1 + DUMMY_RATE)))   # the number of real nodes plus dummy nodes in cvrp
        self.real_size = p_size # the number of real nodes in cvrp
        self.dummy_size = self.size - self.real_size
        self.do_assert = with_assert
        self.step_method = step_method
        self.init_val_met = init_val_met
        self.state = 'eval'
        self.P = P # for perturb
        print(f'CVRP with {self.real_size} nodes and {self.dummy_size} dummy depot.\n', 
              ' Do assert:', with_assert)
        self.train()
    
    def eval(self, perturb = True):
        self.training = False
        self.do_perturb = perturb
        
    def train(self):
        self.training = True
        self.do_perturb = False
    
    def input_feature_encoding(self, batch):
        return torch.cat((batch['coordinates'], batch['demand'].unsqueeze(-1)), -1) # solution-independent features

    def get_real_mask(self, rec, batch):
        
        # get mixed contex: 1000 * route_plan + 1 * visited_time + 0.5 * cu_demand  
        # (e.g., 1000 + 34 + 0.05 means the node is the 34th node in route 1 and the cum demand before the node is 0.1)
        contex, patial_sum = self.preprocessing(rec, batch)
        
        if self.step_method == '2_opt':
            
            # only allow in-route 2-opt
            route_plan = (contex // 1000).long() % self.dummy_size            
            
            # further allow btw-route 2-opt
            cum_demand = ((contex % 1) * 2)
            
            total = patial_sum.gather(-1, route_plan)
            
            pi = cum_demand.view(-1,self.size,1)
            pj = cum_demand.view(-1,1,self.size) 
            
            ti = total.view(-1,self.size,1)
            tj = total.view(-1,1,self.size)
            
            demand = batch['demand'] if isinstance(batch, dict) else batch[:,:,-1]
            
            
            cor = (demand != 0).float().view(-1,1,self.size) 
            
            corj = demand.view(-1,1,self.size) 
            
            mask = ((pi + pj + corj) > 1.) | ((ti - pi * cor) + (tj - pj * cor) - corj > 1.)
            
            mask[:,:self.dummy_size, :self.dummy_size] = False
        
            return mask, contex, torch.cat((cum_demand.view(-1, self.size, 1),
                                            demand.view(-1, self.size, 1),
                                            (total - cor.view(-1,self.size) * cum_demand).view(-1, self.size, 1),
                                            ), -1
                                            ) 

                                        
        elif self.step_method == 'insert':
            
            route_plan = (contex // 1000).long() % self.dummy_size

            cum_demand = ((contex % 1) * 2)
            
            total = patial_sum.gather(-1, route_plan)
            
            pi = cum_demand.view(-1,self.size,1)
            pj = cum_demand.view(-1,1,self.size) 
            
            ti = total.view(-1,self.size,1)
            tj = total.view(-1,1,self.size)
            
            demand = batch['demand'] if isinstance(batch, dict) else batch[:,:,-1]
            
            
            corri = (demand != 0).float().view(-1,self.size, 1) 
            
            cori = demand.view(-1,self.size,1)
            corj = demand.view(-1,1,self.size) 
            
            
            mask = ((ti - pi * corri)  + pi > 1.) | ( cori + tj  > 1.)
        
            return mask, contex, torch.cat((cum_demand.view(-1, self.size, 1),
                                            demand.view(-1, self.size, 1),
                                            (total - corri.view(-1,self.size) * cum_demand).view(-1, self.size, 1),
                                            ), -1
                                            ) 

        else:
            raise NotImplementedError()
        
        
    
    def get_initial_solutions(self, batch):
        
        batch_size = batch['coordinates'].size(0)
    
        def get_solution(methods):
            p_size = self.size
            
            if methods == 'random':
                
                candidates = torch.ones(batch_size,self.size).bool()
                candidates[:,:self.dummy_size] = False
                
                rec = torch.zeros(batch_size, self.size).long()
                selected_node = torch.zeros(batch_size, 1).long()
                cum_demand = torch.zeros(batch_size, 2)
                
                demand = batch['demand'].cpu()
                selected = []
                
                for i in range(self.size - 1):
                    
                    if True:
                        dists = torch.arange(p_size).view(-1, p_size).expand(batch_size, p_size).clone()
                    else:
                        dists = torch.rand(batch_size, p_size)
                    
                    dists.scatter_(1, selected_node, 1e5)
                    dists[~candidates] = 1e5
                    
                    dists[cum_demand[:,-1:] + demand > 1.] = 1e5
                    dists.scatter_(1,cum_demand[:,:-1].long() + 1, 1e4)
                    
                    next_selected_node = dists.min(-1)[1].view(-1,1)
                    selected_demand = demand.gather(1,next_selected_node)
                    cum_demand[:,-1:] = torch.where(selected_demand >0, selected_demand + cum_demand[:,-1:], 0 * cum_demand[:,-1:])
                    cum_demand[:,:-1] = torch.where(selected_demand >0, cum_demand[:,:-1], cum_demand[:,:-1] + 1)
      
                    
                    rec.scatter_(1,selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node
                    selected.append(next_selected_node[-1].item())    
                    
                return rec
            
            
            elif methods == 'greedy':

                candidates = torch.ones(batch_size,self.size).bool()
                candidates[:,:self.dummy_size] = False
                
                rec = torch.zeros(batch_size, self.size).long()
                selected_node = torch.zeros(batch_size, 1).long()
                cum_demand = torch.zeros(batch_size, 2)
                
                d2 = batch['coordinates'].cpu()
                demand = batch['demand'].cpu()
                selected = []
                
                for i in range(self.size - 1):
                    
                    d1 = batch['coordinates'].cpu().gather(1, selected_node.unsqueeze(-1).expand(batch_size, self.size, 2))
                    
                    dists = (d1 - d2).norm(p=2, dim=2)
                    
                    dists.scatter_(1, selected_node, 1e5)
                    dists[~candidates] = 1e5
                    
                    dists[cum_demand[:,-1:] + demand > 1.] = 1e5
                    dists.scatter_(1,cum_demand[:,:-1].long() + 1, 1e4)
                    
                    next_selected_node = dists.min(-1)[1].view(-1,1)
                    selected_demand = demand.gather(1,next_selected_node)
                    cum_demand[:,-1:] = torch.where(selected_demand >0, selected_demand + cum_demand[:,-1:], 0 * cum_demand[:,-1:])
                    cum_demand[:,:-1] = torch.where(selected_demand >0, cum_demand[:,:-1], cum_demand[:,:-1] + 1)
      
                    
                    rec.scatter_(1,selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node
                    selected.append(next_selected_node[-1].item())                            

                return rec
            
            else:
                raise NotImplementedError()

        return get_solution(self.init_val_met).expand(batch_size, self.size).clone()
    
    def step(self, batch, rec, exchange, pre_bsf, solving_state = None, best_solution = None):

        bs = exchange.size(0)
        pre_bsf = pre_bsf.view(bs,-1)
        
        first = exchange[:,0].view(bs,1)
        second = exchange[:,1].view(bs,1)
        
        if self.step_method  == 'swap':
            next_state = self.swap(rec, first, second)
        elif self.step_method  == '2_opt':            
            next_state = self.two_opt(rec, first, second)
        elif self.step_method  == 'insert':
            next_state = self.insert(rec, first, second)
        elif self.step_method  == 'crossover':
            next_state = self.crossover(rec, first, second)
        else:
            raise NotImplementedError()
        
        new_obj = self.get_costs(batch, next_state)
        
        now_bsf = torch.min(torch.cat((new_obj[:,None], pre_bsf[:,-1, None]),-1),-1)[0]
        
        reward = pre_bsf[:,-1] - now_bsf
        
        # update solving state
        solving_state[:,:1] = (1 - (reward > 0).view(-1,1).long()) * (solving_state[:,:1] + 1)
        
        
        if self.do_perturb:
            
            perturb_index = (solving_state[:,:1] > self.P).view(-1)
            
            solving_state[:,:1][perturb_index.view(-1, 1)] *= 0
            
            pertrb_cnt = perturb_index.sum().item()
            
            if pertrb_cnt > 0:
                next_state[perturb_index] =  best_solution[perturb_index]
            
            return next_state, reward, torch.cat((new_obj[:,None], now_bsf[:,None]),-1) , solving_state
        
        return next_state, reward, torch.cat((new_obj[:,None], now_bsf[:,None]),-1) , solving_state
        
    
    def insert(self, solution, first, second, is_perturb = False): # insert first to the back of second
        
        rec = solution.clone()
        
        # fix connection for first node
        argsort = solution.argsort()
        
        pre_first = argsort.gather(1,first)
        post_first = solution.gather(1,first)
        
        rec.scatter_(1,pre_first,post_first)
        
        # fix connection for second node
        post_second = rec.gather(1,second)
        
        rec.scatter_(1,second, first)
        rec.scatter_(1,first, post_second)
        
        return rec
    
    def two_opt(self, solution, first, second, is_perturb = False):
        
        rec = solution.clone()
        
        # fix connection for first node
        argsort = solution.argsort()
        pre_first = argsort.gather(1,first)  
        pre_first = torch.where(pre_first != second, pre_first, first)
        rec.scatter_(1,pre_first,second)
        
        # fix connection for second node
        post_second = solution.gather(1,second)
        post_second = torch.where(post_second != first, post_second, second)
        rec.scatter_(1,first, post_second)
        
        # reverse loop:
        cur = first
        for i in range(self.size):
            cur_next = solution.gather(1,cur)
            rec.scatter_(1,cur_next, torch.where(cur != second,cur,rec.gather(1,cur_next)))
            cur = torch.where(cur != second, cur_next, cur)
        
        return rec
    
    
    def swap(self, solution, first_, second_, is_perturb = False):
        """
        :param solution: [batch_size, size]
        :param first_: [batch_size, 1]
        :param second_: [batch_size, 1]
        :param is_perturb:
        :return:
        """
        con = solution.gather(1,second_) == first_
        first = torch.where(con, second_, first_)
        second = torch.where(con, first_, second_)

        argsort = solution.argsort()
        pre_first = argsort.gather(1,first)
      
        # put first behind second   
        rec1 = self.insert(solution, first, second, is_perturb)            
        # put second behind pre_first 
        rec = self.insert(rec1, second, pre_first, is_perturb)
        
        return rec

    def insert_by_index(self, solution, index1, index2, is_perturb=False, need_protect=False, protect_start_index=None):
        """insert element of index1 to index2
        :param solution:
        :param index1
        :param index2
        :param is_perturb
        :param need_protect
        :param protect_start_index
        :return:
        """
        rec = solution.clone()
        first = rec.gather(1, index1)
        shape = index1.size()

        # shift left
        for i in range(self.size-1):
            index_i = torch.full(shape, i)
            index_i1 = torch.full(shape, i + 1)
            need_shift_left = torch.logical_and(index1 <= index_i, index_i < index2)
            if need_protect:
                need_shift_left = torch.logical_and(need_shift_left, i < protect_start_index)
                index_i1 = torch.where(i + 1 < protect_start_index, index_i1, index2)
            rec.scatter_(1, index_i, torch.where(need_shift_left, rec.gather(1, index_i1), rec.gather(1, index_i)))

        # shift right
        for i in range(self.size - 1, 0, -1):
            index_i = torch.full(shape, i)
            index_i1 = torch.full(shape, i - 1)
            need_shift_right = torch.logical_and(index1 >= index_i, index_i > index2)
            rec.scatter_(1, index_i, torch.where(need_shift_right, rec.gather(1, index_i1), rec.gather(1, index_i)))

        # insert
        rec.scatter_(1, index2, first)

        return rec


    def crossover(self, solution1_, solution2_, index1, index2, is_perturb=False):
        """
        :param solution1_: [batch_size, size]
        :param solution2_: [batch_size, size]
        :param index1: [batch_size, 1]
        :param index2: [batch_size, 1]
        :param is_perturb:
        :return:
        """
        rec1 = solution1_.clone()
        rec2 = solution2_.clone()
        argsort1 = rec1.argsort()
        argsort2 = rec2.argsort()

        shape = index1.size()

        for i in range(self.size):
            index_i = torch.full(shape, i).long()
            con = torch.logical_and(index1 <= i, i <= index2)

            index1_ = torch.where(con, argsort1.gather(1, solution2_.gather(1, index_i)), index_i)
            index2_ = torch.where(con, argsort2.gather(1, solution1_.gather(1, index_i)), index_i)

            rec1 = self.insert_by_index(rec1, index1_, index_i, False, True, index1)
            rec2 = self.insert_by_index(rec2, index2_, index_i, False, True, index1)
            argsort1 = rec1.argsort()
            argsort2 = rec2.argsort()

        return rec1, rec2


    # def crossover(self, solution1_, solution2_, index1, index2, is_perturb=False):
    #     """
    #     :param solution1_: [batch_size, size]
    #     :param solution2_: [batch_size, size]
    #     :param index1: [batch_size, 1]
    #     :param index2: [batch_size, 1]
    #     :param is_perturb:
    #     :return:
    #     """
    #     batch_size = solution1_.size(0)
    #
    #     solution1, solution2 = None, None
    #     for i in range(batch_size):
    #         new_c1, new_c2 = self.single_crossover(solution1_[i], solution2_[i], index1[i], index2[i])
    #         if i == 0:
    #             solution1, solution2 = new_c1, new_c2
    #         else:
    #             solution1 = torch.cat((solution1, new_c1), 0)
    #             solution2 = torch.cat((solution2, new_c2), 0)
    #
    #     return solution1, solution2


    # def single_crossover(self, f1, f2, cro1_index, cro2_index):
    #     new_c1_f, new_c1_m, new_c1_b = torch.empty(0), f1[cro1_index:cro2_index + 1], torch.empty(0)
    #     new_c2_f, new_c2_m, new_c2_b = torch.empty(0), f2[cro1_index:cro2_index + 1], torch.empty(0)
    #     cnt1, cnt2 = 0, 0
    #     for index in range(self.size):
    #         if cnt1 < cro1_index:
    #             if f2[index] not in new_c1_m:
    #                 new_c1_f = torch.cat((new_c1_f, f2[index].expand(1)), 0)
    #                 cnt1 += 1
    #         else:
    #             if f2[index] not in new_c1_m:
    #                 new_c1_b = torch.cat((new_c1_b, f2[index].expand(1)), 0)
    #     for index in range(self.size):
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


    def check_feasibility(self, rec, batch):
        
        p_size = self.size

        assert (
            (torch.arange(p_size, out=rec.new())).view(1, -1).expand_as(rec)  == 
            rec.sort(1)[0]
        ).all(), "not visiting all nodes"
        
        partial_sum = self.preprocessing(rec, batch, True)[-1]
        
        assert (partial_sum <= 1 + 1e-5).all(), ("not satisfying capacity constraint")
    
    
    def get_swap_mask(self, rec, batch):
        
        bs, gs = rec.size()        
        selfmask = torch.eye(gs, device = rec.device).view(1,gs,gs)
        
        real_mask, contex, to_actor = self.get_real_mask(rec, batch)
        masks = (real_mask) + selfmask.expand(bs,gs,gs).bool()
        
        return masks, contex, to_actor
        
    def get_costs(self, batch, rec):
        
        batch_size, size = rec.size()
        
        # check feasibility
        if self.do_assert:
            self.check_feasibility(rec, batch)
        
        # calculate obj value
        #first_row = torch.arange(size, device = rec.device).long().unsqueeze(0).expand(batch_size, size)
        
        d1 = batch['coordinates'].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
        d2 = batch['coordinates']#.gather(1, first_row.unsqueeze(-1).expand(batch_size, size, 2))
        length =  (d1  - d2).norm(p=2, dim=2).sum(1)
        
        return length
    
    def preprocessing(self, solutions, batch, req_partial_sum = True):
        
        batch_size, seq_length = solutions.size()
        demand = batch['demand'] if isinstance(batch, dict) else batch[:,:,-1]
        
        pre = torch.zeros(batch_size, device = solutions.device).long()
        route = torch.zeros(batch_size, device = solutions.device)
        
        route_plan1000_visited_time1_dot_demand = torch.zeros((batch_size,seq_length), device = solutions.device)
            
        if req_partial_sum:
            partial_sum = torch.zeros((batch_size, self.dummy_size), device = solutions.device)
        else:
            partial_sum = None
        
        for i in range(seq_length):
            next_ = solutions[torch.arange(batch_size),pre]
            route = torch.where(next_ < self.dummy_size, route + 1, route)
            
            cu_demand = partial_sum[torch.arange(batch_size),route.long() % self.dummy_size]
            cu_demand = torch.where(next_ < self.dummy_size, partial_sum[torch.arange(batch_size),(route.long() - 1) % self.dummy_size], partial_sum[torch.arange(batch_size),route.long() % self.dummy_size])
            
            route_plan1000_visited_time1_dot_demand[torch.arange(batch_size),next_] = i+1 + route * 1000 + cu_demand * 0.5
            pre = next_
            if req_partial_sum:
               partial_sum[torch.arange(batch_size),route.long() % self.dummy_size] += demand.gather(1,next_.view(-1,1)).view(-1)

        return route_plan1000_visited_time1_dot_demand, partial_sum
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        return CVRPDataset(*args, **kwargs)


class CVRPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0, distribution=None, DUMMY_RATE = None):
        
        super(CVRPDataset, self).__init__()
        
        # From VRP with RL paper https://arxiv.org/abs/1802.04240
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.
        }
        
        self.data = []
        self.size = int(np.ceil(size * (1 + DUMMY_RATE))) # the number of real nodes plus dummy nodes in cvrp
        self.real_size = size # the number of real nodes in cvrp

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [self.make_instance(args) for args in data[offset:offset+num_samples]]

        else:            
            self.data = [{'coordinates': torch.cat((torch.FloatTensor(1, 2).uniform_(0, 1).repeat(self.size - self.real_size,1), 
                                                    torch.FloatTensor(self.real_size, 2).uniform_(0, 1)), 0),
                          'demand': torch.cat((torch.zeros(self.size - self.real_size),
                                               torch.FloatTensor(self.real_size).uniform_(1, 10).long() / CAPACITIES[self.real_size]), 0)
                          } for i in range(num_samples)]
            
            
        
        self.N = len(self.data)
        print(f'{self.N} instances initialized.')
    
    def make_instance(self, args):
        depot, loc, demand, capacity, *args = args
        
        depot = torch.FloatTensor(depot)
        loc = torch.FloatTensor(loc)
        demand = torch.FloatTensor(demand)
        
        return {'coordinates': torch.cat((depot.view(-1, 2).repeat(self.size - self.real_size,1), loc), 0),
                'demand': torch.cat((torch.zeros(self.size - self.real_size), demand / capacity), 0) }
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]


def get_real_seq(solutions):
    batch_size, seq_length = solutions.size()
    visited_time = torch.zeros((batch_size,seq_length))
    pre = torch.zeros((batch_size),device = solutions.device).long()
    for i in range(seq_length):
       visited_time[torch.arange(batch_size),solutions[torch.arange(batch_size),pre]] = i+1
       pre = solutions[torch.arange(batch_size),pre]
       
    visited_time = visited_time % seq_length
    return visited_time.argsort()  
