import torch
import torch_geometric.transforms as T
from torch_geometric.nn import radius_graph
from torch_geometric.utils import coalesce

# Indices: [Pos(0-3), Vel(3-6), Type(6-8), Stress(8-9)]
IDX_VEL = slice(3, 6)

class OverwriteKinematicVelocity(T.BaseTransform):
    def forward(self, data):
        # 6 is index for Sphere Type (One-Hot [1, 0])
        sphere_mask = (data.x[:, 6] > 0.5)
        
        if sphere_mask.sum() > 0:
            # Overwrite input velocity with target (future) velocity
            v_future = data.y[:, 0:3]
            data.x[sphere_mask, IDX_VEL] = v_future[sphere_mask]
            
        return data

class AddDynamicWorldEdges(T.BaseTransform):
    def __init__(self, radius=0.03):
        self.radius = radius

    def forward(self, data):
        # Use Pos (first 3 cols) for radius search
        pos = data.x[:, 0:3] 
        batch = data.batch if 'batch' in data else None
        
        new_edges = radius_graph(pos, r=self.radius, batch=batch, loop=False)
        
        if new_edges.numel() > 0:
            data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
            data.edge_index = coalesce(data.edge_index, num_nodes=data.num_nodes)
        return data