import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from graphviz import Digraph

from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models


# def make_dot(var):
#     node_attr = dict(style='filled',
#                      shape='box',
#                      align='left',
#                      fontsize='12',
#                      ranksep='0.1',
#                      height='0.2')
#     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
#     seen = set()

#     def add_nodes(var):
#         if var not in seen:
#             if isinstance(var, Variable):
#                 value = '('+(', ').join(['%d'% v for v in var.size()])+')'
#                 dot.node(str(id(var)), str(value), fillcolor='lightblue')
#             else:
#                 dot.node(str(id(var)), str(type(var).__name__))
#             seen.add(var)
#             if hasattr(var, 'previous_functions'):
#                 for u in var.previous_functions:
#                     dot.edge(str(id(u[0])), str(id(var)))
#                     add_nodes(u[0])
#     add_nodes(var.creator)
#     return dot


# inputs = torch.randn(1,3,224,224)
# resnet18 = models.resnet18()
# y = resnet18(Variable(inputs))
# # print(y)

# g = make_dot(y)
# g.view()

class IQMMLP(nn.Module):
    def __init__(self, n_features=65, n_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 50, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(50, 20, bias=True),
            nn.BatchNorm1d(20),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),            
            
            nn.Linear(20, n_classes)
        )
        # initialize the weights in the model
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.model(x)





if __name__=='__main__':
    raw_data = pd.read_csv('abide.csv', header=None)
    raw_data = raw_data.as_matrix()
    X = raw_data[:,:-1]
    X = X.astype(np.float32)
    y = raw_data[:,-1]
    y = y.astype(np.int64)

    model = nn.Sequential()
    model.add_module('W_0', nn.Linear(65, 50))
    model.add_module('ReLU_0', nn.ReLU(inplace=True))
    model.add_module('BatchNorm_0', nn.BatchNorm1d(50))
    model.add_module('Dropout_0', nn.Dropout(0.25))

    model.add_module('W_1', nn.Linear(50, 20))
    model.add_module('ReLU_1', nn.ReLU(inplace=True))
    model.add_module('BatchNorm_1', nn.BatchNorm1d(20))
    model.add_module('Dropout_1', nn.Dropout(0.25))

    model.add_module('W_2', nn.Linear(20, 2))

    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    print(model)

    x = torch.randn(5,65)
    x = Variable(x)
    make_dot(model(x), params=model.named_parameters())
    