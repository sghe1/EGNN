import torch
import torch_geometric.transforms as T
from torch_geometric.nn import radius_graph
from torch_geometric.utils import coalesce

# Indices based on preprocess: [Pos(0-3), Vel(3-6), Type(6-8), Stress(8-9)]
IDX_VEL = slice(3, 6)
SPHERE_NODE = 1

class OverwriteKinematicVelocity(T.BaseTransform):
    """
    Implements the logic:
    If Node is Sphere: Input Velocity = Target Velocity (t+1)
    If Node is Plate:  Input Velocity = Current Velocity (t)
    """
    def forward(self, data):
        # 1. Identify Sphere Nodes
        sphere_mask = (data.node_type == SPHERE_NODE).squeeze()
        
        if sphere_mask.sum() > 0:
            # 2. Get Future Velocity from Target (y)
            # data.y is [Vel(3), Stress(1)] -> take first 3
            v_future = data.y[:, 0:3]
            
            # 3. Overwrite Input Velocity (x)
            # data.x indices 3:6 is Velocity
            data.x[sphere_mask, IDX_VEL] = v_future[sphere_mask]
            
        return data

class AddDynamicWorldEdges(T.BaseTransform):
    """Adds collision edges based on radius search"""
    def __init__(self, radius=0.03):
        self.radius = radius

    def forward(self, data):
        pos = data.x[:, 0:3] # World Pos is first 3
        batch = data.batch if 'batch' in data else None
        
        # Radius Search
        new_edges = radius_graph(pos, r=self.radius, batch=batch, loop=False)
        
        if new_edges.numel() > 0:
            data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
            data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)
        return data