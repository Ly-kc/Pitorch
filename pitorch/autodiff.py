"""
本文件我们给出进行自动微分的步骤
你可以将lab5的对应代码复制到这里
"""

import numpy as np
from typing import List, Dict, Tuple
from basic_operator import Op, Value

def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """
    给定一个节点列表，返回以这些节点结束的拓扑排序列表。
    一种简单的算法是对给定的节点进行后序深度优先搜索（DFS）遍历，
    根据输入边向后遍历。由于一个节点是在其所有前驱节点遍历后才被添加到排序中的，
    因此我们得到了一个拓扑排序。
    """
    topo_order = []
    visited = set()
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)

    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if(node in visited):
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    
    topo_order.append(node)
    

def compute_gradient_of_variables(output_tensor, out_grad):
    """
    对输出节点相对于 node_list 中的每个节点求梯度。
    将计算结果存储在每个 Variable 的 grad 字段中。
    """
    # map for 从节点到每个输出节点的梯度贡献列表
    node_to_output_grads_list = {}
    # 我们实际上是在对标量 reduce_sum(output_node) 
    # 而非向量 output_node 取导数。
    # 但这是损失函数的常见情况。
    node_to_output_grads_list[output_tensor] = [out_grad]

    # 根据我们要对其求梯度的 output_node，以逆拓扑排序遍历图。
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    raise NotImplementedError()
            
            
def get_node_set(output_tensor):
    node_set = set()
    node_queue = []
    node_queue.append(output_tensor)
    while(len(node_queue) != 0):
        node = node_queue.pop(0)
        if(node.dirty):
            raise Exception('inplaced node found in computing graph, backward failed')
        if(node not in node_set):
            node_set.add(node)
            for n in node.inputs:
                node_queue.append(n)
    return node_set


def print_compute_graph(rev_topo_node_list):
    for node in rev_topo_node_list:
        print([input.tensor_number for input in node.inputs], node.tensor_number, node.op)
        
    
'''
only support one order of gradient now...
'''
def back_propgation(output_tensor, out_grad):
    
    from Pisor import pisor
    
    node_to_grad = {}
    
    node_list = list(get_node_set(output_tensor))
    for node in node_list:
        node_to_grad[node] = []
    if(len(output_tensor.shape)==0):
        node_to_grad[output_tensor] = [pisor(np.array(1), device=output_tensor.device)]  #相当于反向的输入，不需要记录梯度
    elif(len(output_tensor.shape)==1):
        assert(output_tensor.shape[0] == 1)
        node_to_grad[output_tensor] = [pisor(np.array([1]), device=output_tensor.device)]
    else:
        assert(output_tensor.shape == out_grad.shape)
        node_to_grad[output_tensor] = [out_grad]
        
    rev_topo_node_list = list(reversed(find_topo_sort(node_list)))
    
    '''
    Without the code below, TENSOR_NUMBER will burst. How magical...
    '''
    for node in rev_topo_node_list:
        node.grad = None
    
    for onode in rev_topo_node_list:
        if(onode.requires_grad == False):
            onode.grad = None
            continue
        #为所有的输入节点积累梯度
        if(onode.grad is None):
            onode.grad = node_to_grad[onode][0]
        else:
            onode.grad = onode.grad + node_to_grad[onode][0]  
        for grad in node_to_grad[onode][1:]:
            onode.grad = onode.grad + grad

        if(onode.grad.shape != onode.shape):
            raise Exception(f'shape not match: {onode.grad.shape} {onode.shape}')

        if(onode.is_leaf()): continue
        partial_gradients = onode.partial_gradients()
        for i,snode in enumerate(onode.inputs):
            node_to_grad[snode].append(partial_gradients[i])
    