import torch
import torch.nn as nn
import torch.nn.functional as F
# from audtorch.metrics.functional import pearsonr

class Loss(nn.Module):
    def __init__(self, t, Nlabel, device):
        super(Loss, self).__init__()

        self.Nlabel = Nlabel
        self.t = t
        self.device = device
        self.CE = nn.CrossEntropyLoss(reduction="sum")
        self.mse = nn.MSELoss()


    def label_graph2(self, emb, inc_labels, inc_L_ind):
        # label guide the embedding feature
        cls_num = inc_labels.shape[-1]
        valid_labels_sum = torch.matmul(inc_L_ind.float(), inc_L_ind.float().T) #[n, n] 

        # graph = torch.matmul(inc_labels, inc_labels.T).fill_diagonal_(0)
        graph = (torch.matmul(inc_labels, inc_labels.T).mul(valid_labels_sum) / (torch.matmul(inc_labels, inc_labels.T).mul(valid_labels_sum)+100)).fill_diagonal_(0)
        # print((graph>0.1).sum(),graph.shape)
        # assert torch.sum(torch.isnan(graph)).item() == 0
        graph = torch.clamp(graph,min=0,max=1.)
        emb = F.normalize(emb, p=2, dim=-1)
        # graph = graph.mul(graph>0.2)
        # graph = (inc_labels.mm(inc_labels.T))
        # graph = 0.5*(graph+graph.t())¸
        
        loss = 0
        Lap_graph  = torch.diag(graph.sum(1))- graph
        loss = torch.trace(emb.t().mm(Lap_graph).mm(emb))/emb.shape[0]
        return loss/emb.shape[0] #loss/number of views




    def wmse_loss(self,input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        ret = torch.mean(ret)
        return ret

    def cont_loss(self,S,V,inc_V_ind):
        loss_Cont = 0
        if isinstance(S,list):
            S = torch.stack(S,1) #[n v d]

        if isinstance(V,list):
            V = torch.stack(V,1) #[n v d]
        for i in range(S.size(0)):
            loss_Cont += self.forward_contrast(S[i], V[i], inc_V_ind[i,:])
        return loss_Cont
    def forward_contrast(self, si, vi, wei):
        ## S1 S2 [v d]
        si = si[wei.bool()]
        vi = vi[wei.bool()]
        n = si.size(0)
        N = 2 * n
        if n <= 1:
            return 0
        si = F.normalize(si, p=2, dim=1)
        vi = F.normalize(vi, p=2, dim=1)
        if si.shape[0]<=1 and vi.shape[0]<=1:
            return 0

        svi = torch.cat((si, vi), dim=0)

        sim = torch.matmul(svi, svi.T)
        # sim = (sim/self.t).exp()
        # print(sim)

        pos_mask = torch.zeros((N, N),device=sim.device)
        pos_mask[:n,:n] = torch.ones((n, n),device=sim.device)
        neg_mask = 1-pos_mask
        pos_mask = pos_mask.fill_diagonal_(0)
        neg_mask = neg_mask.fill_diagonal_(0)
        pos_pairs = sim.masked_select(pos_mask.bool())
        neg_pairs = sim.masked_select(neg_mask.bool())
        # prop = torch.exp(pos_pairs).mean()/(torch.exp(pos_pairs).mean()+torch.abs(torch.exp(neg_pairs)).mean())
        # loss = -torch.log(prop)
        loss = (neg_pairs).square().mean()/(((pos_pairs+1+1e-6)/2).mean())
        # loss = (neg_pairs).square().mean()/(pos_pairs).square().mean()
        # target = torch.eye(N,device=sim.device)
        # target[:n,:n] = torch.ones((n, n),device=sim.device)
        # loss = (-target.mul(torch.log((sim+1)/2+1e-6))-(1-target).mul(torch.log(1-sim.square()+1e-6))).mean()

        assert torch.sum(torch.isnan(loss)).item() == 0
        return loss/2

   
    
 
    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0

        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        
        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res



    
