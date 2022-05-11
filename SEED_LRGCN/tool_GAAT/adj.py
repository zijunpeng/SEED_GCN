import numpy as np
import scipy.sparse as sp
import torch

def draw_adj():
    """
    draw double-side adjacency matrix
    """
    adj = []

    adj.extend([[0,1],[0,7]])
    adj.extend([[1,0],[1,2],[1,8]]) 
    adj.extend([[2,1],[2,3],[2,9]])
    adj.extend([[3,2],[3,4],[3,10]])
    adj.extend([[4,3],[4,11]])

    adj.extend([[5,6],[5,14]])
    adj.extend([[6,5],[6,7],[6,15]])
    adj.extend([[7,0],[7,6],[7,8],[7,16]])
    adj.extend([[8,1],[8,7],[8,9],[8,17]])
    adj.extend([[9,2],[9,8],[9,10],[9,18]])
    adj.extend([[10,3],[10,9],[10,11],[10,19]])
    adj.extend([[11,4],[11,10],[11,12],[11,20]])
    adj.extend([[12,11],[12,13],[12,21]])
    adj.extend([[13,12],[13,22]])

    adj.extend([[14,5],[14,15],[14,23]])
    adj.extend([[15,6],[15,14],[15,16],[15,24]])
    adj.extend([[16,7],[16,15],[16,17],[16,25]])
    adj.extend([[17,8],[17,16],[17,18],[17,26]])
    adj.extend([[18,9],[18,17],[18,19],[18,27]])
    adj.extend([[19,10],[19,18],[19,20],[19,28]])
    adj.extend([[20,11],[20,19],[20,21],[20,29]])
    adj.extend([[21,12],[21,20],[21,22],[21,30]])
    adj.extend([[22,13],[22,21],[22,31]])

    adj.extend([[23,14],[23,24],[23,32]])
    adj.extend([[24,15],[24,23],[24,25],[24,33]])
    adj.extend([[25,16],[25,24],[25,26],[25,34]])
    adj.extend([[26,17],[26,25],[26,27],[26,35]])
    adj.extend([[27,18],[27,26],[27,28],[27,36]])
    adj.extend([[28,19],[28,27],[28,29],[28,37]])
    adj.extend([[29,20],[29,28],[29,30],[29,38]])
    adj.extend([[30,21],[30,29],[30,31],[30,39]])
    adj.extend([[31,22],[31,30],[31,40]])

    adj.extend([[32,23],[32,33],[32,41]])
    adj.extend([[33,24],[33,32],[33,34],[33,42]])
    adj.extend([[34,25],[34,33],[34,35],[34,43]])
    adj.extend([[35,26],[35,34],[35,36],[35,44]])
    adj.extend([[36,27],[36,35],[36,37],[36,45]])
    adj.extend([[37,28],[37,36],[37,38],[37,46]])
    adj.extend([[38,29],[38,37],[38,39],[38,47]])
    adj.extend([[39,30],[39,38],[39,40],[39,48]])
    adj.extend([[40,31],[40,39],[40,49]])

    adj.extend([[41,32],[41,42]])
    adj.extend([[42,33],[42,41],[42,43],[42,51]])
    adj.extend([[43,34],[43,42],[43,44],[43,52]])
    adj.extend([[44,35],[44,43],[44,45],[44,53]])
    adj.extend([[45,36],[45,44],[45,46],[45,54]])
    adj.extend([[46,37],[46,45],[46,47],[46,55]])
    adj.extend([[47,38],[47,46],[47,48],[47,56]])
    adj.extend([[48,39],[48,47],[48,49],[48,57]])
    adj.extend([[49,40],[49,48]])

    adj.extend([[50,42],[50,51]])
    adj.extend([[51,43],[51,50],[51,52],[51,57]])
    adj.extend([[52,44],[52,51],[52,53],[52,58]])
    adj.extend([[53,45],[53,52],[53,54],[53,59]])
    adj.extend([[54,46],[54,53],[54,55],[54,60]])
    adj.extend([[55,47],[55,54],[55,56],[55,61]])
    adj.extend([[56,48],[56,55]])

    adj.extend([[57,51],[57,58]])
    adj.extend([[58,52],[58,57],[58,59]])
    adj.extend([[59,53],[59,58],[59,60]])
    adj.extend([[60,54],[60,59],[60,61]])
    adj.extend([[61,55],[61,60]])

    adj = np.array(adj)
    adj = sp.coo_matrix((np.ones(adj.shape[0]), (adj[:, 0], adj[:, 1])),
                         shape=(62, 62),
                         dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = sp.lil_matrix(adj).toarray()
    adj = torch.tensor(adj)

    
    return adj 


def normalize_adj(adj):
    """
    L = I + D^(-1/2)A'D^(-1/2)
    """
    I = torch.FloatTensor(torch.eye(adj.size()[0]))
    I = I.cuda()
    A = torch.add(adj, I) # A' = A + I

    D = torch.diag(torch.sum(A, dim=1))
    D_ = torch.diag(torch.diag(1 / torch.sqrt(D))) # D^(-1/2)
    L = torch.matmul(D_, torch.matmul(A, D_))
    
    return L






