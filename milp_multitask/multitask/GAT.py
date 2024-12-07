import os

import torch
import torch_geometric
from torch_scatter import scatter_mean
import gzip
import pickle
import random
import numpy as np
import time
import pyscipopt as scp
import gurobipy as gp
from IPython import embed
import torch.nn.init as init

class GATPolicy_multitask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4#MIPDataset.num_con_features
        edge_nfeats = 1
        var_nfeats = 15#MIPDataset.num_var_features
        config_nfeats = 33

        # Constraint embedding
        self.cons_norm = Prenorm(cons_nfeats)
        self.cons_embedding = torch.nn.Sequential(
            self.cons_norm,
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # Edge embedding
        self.edge_norm = Prenorm(edge_nfeats)
        self.edge_embedding = torch.nn.Sequential(
            self.edge_norm,
            torch.nn.Linear(edge_nfeats, emb_size),
        )

        # Variable embedding
        self.var_norm = Prenorm(var_nfeats, preserve_features=[2])
        self.var_embedding = torch.nn.Sequential(
            self.var_norm,
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = GATConvolution()
        self.conv_c_to_v = GATConvolution()  
        
        self.output_module1 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.output_module2 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.reset_parameters()


    def reset_parameters(self):
        for t in self.parameters():
            if len(t.shape) == 2:
                init.orthogonal_(t)
            else:
                init.normal_(t)

    def freeze_normalization(self):
        if not self.cons_norm.frozen:
            self.cons_norm.freeze_normalization()
            self.edge_norm.freeze_normalization()
            self.var_norm.freeze_normalization()
            self.conv_v_to_c.reset_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_v_to_c.frozen:
            self.conv_v_to_c.freeze_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_c_to_v.frozen:
            self.conv_c_to_v.freeze_normalization()
            return False
        return True


    def forward(self, constraint_features, edge_indices, edge_features, variable_features, task_id, variable_features_batch=None, constraint_features_batch=None):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)
 
        # A final MLP on the variable features
        if task_id == 1:
            output = self.output_module1(variable_features).squeeze(-1)
        elif task_id == 2:
            output = self.output_module2(variable_features).squeeze(-1)

        return output

class GATPolicy_multitask_v2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4#MIPDataset.num_con_features
        edge_nfeats = 1
        var_nfeats = 15#MIPDataset.num_var_features
        config_nfeats = 33

        # Constraint embedding
        self.cons_norm = Prenorm(cons_nfeats)
        self.cons_embedding = torch.nn.Sequential(
            self.cons_norm,
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # Edge embedding
        self.edge_norm = Prenorm(edge_nfeats)
        self.edge_embedding = torch.nn.Sequential(
            self.edge_norm,
            torch.nn.Linear(edge_nfeats, emb_size),
        )

        # Variable embedding
        self.var_norm = Prenorm(var_nfeats, preserve_features=[2])
        self.var_embedding = torch.nn.Sequential(
            self.var_norm,
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = GATConvolution()
        self.conv_c_to_v = GATConvolution()  
        
        self.output_module1 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.output_module2 = torch.nn.Sequential(
            torch.nn.Linear(emb_size * 2, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, config_nfeats, bias=False),
        )

        # self.output_module3 = torch.nn.Sequential(
        #     torch.nn.Linear(emb_size, emb_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(emb_size, 1, bias=False),
        # )

        # self.output_module4 = torch.nn.Sequential(
        #     torch.nn.Linear(emb_size * 2, emb_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(emb_size, config_nfeats, bias=False),
        # )

        # self.output_module5 = torch.nn.Sequential(
        #     torch.nn.Linear(emb_size, emb_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(emb_size, 1, bias=False),
        # )

        # self.output_module6 = torch.nn.Sequential(
        #     torch.nn.Linear(emb_size * 2, emb_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(emb_size, config_nfeats, bias=False),
        # )

        self.reset_parameters()


    def reset_parameters(self):
        for t in self.parameters():
            if len(t.shape) == 2:
                init.orthogonal_(t)
            else:
                init.normal_(t)

    def freeze_normalization(self):
        if not self.cons_norm.frozen:
            self.cons_norm.freeze_normalization()
            self.edge_norm.freeze_normalization()
            self.var_norm.freeze_normalization()
            self.conv_v_to_c.reset_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_v_to_c.frozen:
            self.conv_v_to_c.freeze_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_c_to_v.frozen:
            self.conv_c_to_v.freeze_normalization()
            return False
        return True


    def forward(self, constraint_features, edge_indices, edge_features, variable_features, task_id, variable_features_batch=None, constraint_features_batch=None):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)
 
        # A final MLP on the variable features
        if task_id == 1:
            output = self.output_module1(variable_features).squeeze(-1)
        elif task_id == 2:
            # Pooling: compute the average of variable and constraint features for each graph in the batch
            pooled_variable = scatter_mean(variable_features, variable_features_batch, dim=0)  # [batch_size, 64]
            pooled_constraint = scatter_mean(constraint_features, constraint_features_batch, dim=0)  # [batch_size, 64]
        
            # Concatenate pooled constraint and variable features for each batch
            pooled_features = torch.cat([pooled_variable, pooled_constraint], dim=-1)  # [batch_size, 128]
        
            # Apply output module to get raw logits, maintaining batch dimension
            raw_output = self.output_module2(pooled_features)  # [batch_size, 33]
        
            # Define groups of indices for softmax (1-3, 8-10, 14-20, 22-28)
            softmax_groups = [
                [1, 2, 3],        # group 1
                [8, 9, 10],       # group 2
                [14, 15, 16, 17, 18, 19, 20],  # group 3
                [22, 23, 24, 25, 26, 27, 28],  # group 4
            ]
            
            # Clone raw_output to final_output to apply softmax to specific groups
            output = raw_output.clone()
        
            # Apply softmax to each group of indices for each batch
            for group in softmax_groups:
                group_output = raw_output[..., group]  # Get the logits for the group across the batch
                softmax_output = torch.softmax(group_output, dim=-1)  # Softmax over the group (last dimension)
                output[..., group] = softmax_output  # Replace the group logits with softmax outputs
        # elif task_id == 3:
        #     output = self.output_module3(variable_features).squeeze(-1)
        # elif task_id == 4:
        #     # Pooling: compute the average of variable and constraint features for each graph in the batch
        #     pooled_variable = scatter_mean(variable_features, variable_features_batch, dim=0)  # [batch_size, 64]
        #     pooled_constraint = scatter_mean(constraint_features, constraint_features_batch, dim=0)  # [batch_size, 64]
        
        #     # Concatenate pooled constraint and variable features for each batch
        #     pooled_features = torch.cat([pooled_variable, pooled_constraint], dim=-1)  # [batch_size, 128]
        
        #     # Apply output module to get raw logits, maintaining batch dimension
        #     raw_output = self.output_module4(pooled_features)  # [batch_size, 33]
        
        #     # Define groups of indices for softmax (1-3, 8-10, 14-20, 22-28)
        #     softmax_groups = [
        #         [1, 2, 3],        # group 1
        #         [8, 9, 10],       # group 2
        #         [14, 15, 16, 17, 18, 19, 20],  # group 3
        #         [22, 23, 24, 25, 26, 27, 28],  # group 4
        #     ]
            
        #     # Clone raw_output to final_output to apply softmax to specific groups
        #     output = raw_output.clone()
        
        #     # Apply softmax to each group of indices for each batch
        #     for group in softmax_groups:
        #         group_output = raw_output[..., group]  # Get the logits for the group across the batch
        #         softmax_output = torch.softmax(group_output, dim=-1)  # Softmax over the group (last dimension)
        #         output[..., group] = softmax_output  # Replace the group logits with softmax outputs
        # elif task_id == 5:
        #     output = self.output_module5(variable_features).squeeze(-1)
        # elif task_id == 6:
        #     # Pooling: compute the average of variable and constraint features for each graph in the batch
        #     pooled_variable = scatter_mean(variable_features, variable_features_batch, dim=0)  # [batch_size, 64]
        #     pooled_constraint = scatter_mean(constraint_features, constraint_features_batch, dim=0)  # [batch_size, 64]
        
        #     # Concatenate pooled constraint and variable features for each batch
        #     pooled_features = torch.cat([pooled_variable, pooled_constraint], dim=-1)  # [batch_size, 128]
        
        #     # Apply output module to get raw logits, maintaining batch dimension
        #     raw_output = self.output_module6(pooled_features)  # [batch_size, 33]
        
        #     # Define groups of indices for softmax (1-3, 8-10, 14-20, 22-28)
        #     softmax_groups = [
        #         [1, 2, 3],        # group 1
        #         [8, 9, 10],       # group 2
        #         [14, 15, 16, 17, 18, 19, 20],  # group 3
        #         [22, 23, 24, 25, 26, 27, 28],  # group 4
        #     ]
            
        #     # Clone raw_output to final_output to apply softmax to specific groups
        #     output = raw_output.clone()
        
        #     # Apply softmax to each group of indices for each batch
        #     for group in softmax_groups:
        #         group_output = raw_output[..., group]  # Get the logits for the group across the batch
        #         softmax_output = torch.softmax(group_output, dim=-1)  # Softmax over the group (last dimension)
        #         output[..., group] = softmax_output  # Replace the group logits with softmax outputs
    
        return output

class GATPolicy_multitask_v3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4#MIPDataset.num_con_features
        edge_nfeats = 1
        var_nfeats = 15#MIPDataset.num_var_features
        config_nfeats = 33

        # Constraint embedding
        self.cons_norm = Prenorm(cons_nfeats)
        self.cons_embedding = torch.nn.Sequential(
            self.cons_norm,
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # Edge embedding
        self.edge_norm = Prenorm(edge_nfeats)
        self.edge_embedding = torch.nn.Sequential(
            self.edge_norm,
            torch.nn.Linear(edge_nfeats, emb_size),
        )

        # Variable embedding
        self.var_norm = Prenorm(var_nfeats, preserve_features=[2])
        self.var_embedding = torch.nn.Sequential(
            self.var_norm,
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = GATConvolution()
        self.conv_c_to_v = GATConvolution()  

        self.output_module1 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.output_module2 = torch.nn.Sequential(
            torch.nn.Linear(emb_size * 2, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, config_nfeats, bias=False),
        )

        self.reset_parameters()


    def reset_parameters(self):
        for t in self.parameters():
            if len(t.shape) == 2:
                init.orthogonal_(t)
            else:
                init.normal_(t)

    def freeze_normalization(self):
        if not self.cons_norm.frozen:
            self.cons_norm.freeze_normalization()
            self.edge_norm.freeze_normalization()
            self.var_norm.freeze_normalization()
            self.conv_v_to_c.reset_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_v_to_c.frozen:
            self.conv_v_to_c.freeze_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_c_to_v.frozen:
            self.conv_c_to_v.freeze_normalization()
            return False
        return True


    def forward(self, constraint_features, edge_indices, edge_features, variable_features, task_id, variable_features_batch=None, constraint_features_batch=None):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)
 
        # A final MLP on the variable features
        if task_id == 1:
            output = self.output_module1(variable_features).squeeze(-1)
        elif task_id == 2:
            # Pooling: compute the average of variable and constraint features for each graph in the batch
            pooled_variable = scatter_mean(variable_features, variable_features_batch, dim=0)  # [batch_size, 64]
            pooled_constraint = scatter_mean(constraint_features, constraint_features_batch, dim=0)  # [batch_size, 64]
        
            # Concatenate pooled constraint and variable features for each batch
            pooled_features = torch.cat([pooled_variable, pooled_constraint], dim=-1)  # [batch_size, 128]
        
            # Apply output module to get raw logits, maintaining batch dimension
            raw_output = self.output_module2(pooled_features)  # [batch_size, 33]
        
            # Define groups of indices for softmax (1-3, 8-10, 14-20, 22-28)
            softmax_groups = [
                [1, 2, 3],        # group 1
                [8, 9, 10],       # group 2
                [14, 15, 16, 17, 18, 19, 20],  # group 3
                [22, 23, 24, 25, 26, 27, 28],  # group 4
            ]
            
            # Clone raw_output to final_output to apply softmax to specific groups
            output = raw_output.clone()
        
            # Apply softmax to each group of indices for each batch
            for group in softmax_groups:
                group_output = raw_output[..., group]  # Get the logits for the group across the batch
                softmax_output = torch.softmax(group_output, dim=-1)  # Softmax over the group (last dimension)
                output[..., group] = softmax_output  # Replace the group logits with softmax outputs

        return output

class GATPolicy_multitask_x3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 4#MIPDataset.num_con_features
        edge_nfeats = 1
        var_nfeats = 15#MIPDataset.num_var_features
        config_nfeats = 33

        # Constraint embedding
        self.cons_norm = Prenorm(cons_nfeats)
        self.cons_embedding = torch.nn.Sequential(
            self.cons_norm,
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # Edge embedding
        self.edge_norm = Prenorm(edge_nfeats)
        self.edge_embedding = torch.nn.Sequential(
            self.edge_norm,
            torch.nn.Linear(edge_nfeats, emb_size),
        )

        # Variable embedding
        self.var_norm = Prenorm(var_nfeats, preserve_features=[2])
        self.var_embedding = torch.nn.Sequential(
            self.var_norm,
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = GATConvolution()
        self.conv_c_to_v = GATConvolution()  
        
        self.output_module1 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.output_module2 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.output_module3 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.output_module4 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.output_module5 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        self.output_module6 = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

        # self.output_module3 = torch.nn.Sequential(
        #     torch.nn.Linear(emb_size * 2, emb_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(emb_size, config_nfeats, bias=False),
        # )

        self.reset_parameters()


    def reset_parameters(self):
        for t in self.parameters():
            if len(t.shape) == 2:
                init.orthogonal_(t)
            else:
                init.normal_(t)

    def freeze_normalization(self):
        if not self.cons_norm.frozen:
            self.cons_norm.freeze_normalization()
            self.edge_norm.freeze_normalization()
            self.var_norm.freeze_normalization()
            self.conv_v_to_c.reset_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_v_to_c.frozen:
            self.conv_v_to_c.freeze_normalization()
            self.conv_c_to_v.reset_normalization()
            return False
        if not self.conv_c_to_v.frozen:
            self.conv_c_to_v.freeze_normalization()
            return False
        return True


    def forward(self, constraint_features, edge_indices, edge_features, variable_features, task_id, variable_features_batch=None, constraint_features_batch=None):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        # Two half convolutions
        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)
 
        # A final MLP on the variable features
        if task_id == 1:
            output = self.output_module1(variable_features).squeeze(-1)
        elif task_id == 2:
            output = self.output_module2(variable_features).squeeze(-1)
        elif task_id == 3:
            output = self.output_module3(variable_features).squeeze(-1)
        elif task_id == 4:
            output = self.output_module4(variable_features).squeeze(-1)
        elif task_id == 5:
            output = self.output_module5(variable_features).squeeze(-1)
        elif task_id == 6:
            output = self.output_module6(variable_features).squeeze(-1)
        # elif task_id == 3:
        #     # Pooling: compute the average of variable and constraint features for each graph in the batch
        #     pooled_variable = scatter_mean(variable_features, variable_features_batch, dim=0)  # [batch_size, 64]
        #     pooled_constraint = scatter_mean(constraint_features, constraint_features_batch, dim=0)  # [batch_size, 64]
        
        #     # Concatenate pooled constraint and variable features for each batch
        #     pooled_features = torch.cat([pooled_variable, pooled_constraint], dim=-1)  # [batch_size, 128]
        
        #     # Apply output module to get raw logits, maintaining batch dimension
        #     raw_output = self.output_module(pooled_features)  # [batch_size, 33]
        
        #     # Define groups of indices for softmax (1-3, 8-10, 14-20, 22-28)
        #     softmax_groups = [
        #         [1, 2, 3],        # group 1
        #         [8, 9, 10],       # group 2
        #         [14, 15, 16, 17, 18, 19, 20],  # group 3
        #         [22, 23, 24, 25, 26, 27, 28],  # group 4
        #     ]
            
        #     # Clone raw_output to final_output to apply softmax to specific groups
        #     output = raw_output.clone()
        
        #     # Apply softmax to each group of indices for each batch
        #     for group in softmax_groups:
        #         group_output = raw_output[..., group]  # Get the logits for the group across the batch
        #         softmax_output = torch.softmax(group_output, dim=-1)  # Softmax over the group (last dimension)
        #         output[..., group] = softmax_output  # Replace the group logits with softmax outputs
    
        return output

class Prenorm(torch.nn.Module):
    def __init__(self, num_features, shift=True, scale=True, preserve_features=[]):
        super().__init__()

        self.num_features = num_features
        self.preserve_features = preserve_features

        self.register_buffer("avg", torch.zeros([num_features], dtype=torch.double))
        self.register_buffer("var", torch.zeros([num_features], dtype=torch.double))
        self.register_buffer("count", torch.zeros([1]))
        self.register_buffer("frozen", torch.tensor([False], dtype=torch.bool, requires_grad=False))

        if shift:
            self.register_buffer("shift", torch.zeros([num_features]))
        else:
            self.shift = None
        if scale:
            self.register_buffer("scale", torch.ones([num_features]))
        else:
            self.scale = None

    def freeze_normalization(self):
        self.frozen = torch.tensor([True], dtype=torch.bool).detach()

    def reset_normalization(self):
        self.avg.zero_()
        self.var.zero_()
        self.count.zero_()
        self.count += 1
        self.frozen.zero_()
        
    def forward(self, input):
        if self.training and not self.frozen:
            # Online mean and variance estimation from Chan et al
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            assert len(input.shape) == 2
            assert self.num_features == input.shape[-1], f"Expected input dimension of size {self.num_features}, got {input.shape[-1]}."

            with torch.no_grad():
                assert not torch.isnan(input).any()
                assert not torch.isnan(self.var).any()
                assert not torch.isnan(self.scale).any()
                assert not torch.isnan(self.count).any()

                sample_count = float(input.shape[0])
                sample_var, sample_avg = torch.var_mean(input.to(torch.float64), dim=0)

                assert not torch.isnan(sample_avg).any()
                assert not torch.isnan(sample_var).any()

                delta = sample_avg - self.avg

                assert self.count + sample_count > 0
                m2 = (self.var * self.count + sample_var * sample_count + torch.square(delta) * self.count * sample_count / (
                    self.count + sample_count))
                assert not torch.isnan(m2).any()

                self.avg = (self.avg * self.count + sample_avg * sample_count) / (self.count + sample_count)
                assert not torch.isnan(self.avg).any()

                self.count += sample_count
                self.var = m2 / self.count

                if self.shift is not None:
                    self.shift = -self.avg.to(torch.float32)
                    assert not torch.isnan(self.shift).any()

                if self.scale is not None:
                    var = torch.where(torch.eq(self.var, 0), self.var.new_ones([self.num_features]), self.var)
                    assert not torch.isnan(var).any()
                    #assert not torch.isinf(var).any()
                    assert (var > 0).all()
                    self.scale = torch.rsqrt(var).to(torch.float32)
                    assert not torch.isnan(self.scale).any()

            for f in self.preserve_features:
                self.shift[f] = 0.0
                self.scale[f] = 1.0

        output = input
        if self.shift is not None:
            output = output + self.shift
        if self.scale is not None:
            output = output * self.scale

        assert not torch.any(torch.isnan(output))

        return output
    

    # GATConvolution network derived https://arxiv.org/abs/2105.14491
# Added edge embedding as well 
class GATConvolution(torch_geometric.nn.MessagePassing):
    """
    Graph convolution layer. THis is the heart of our GNNPolicy
    """
    def __init__(self,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 **kwargs):
        super().__init__('add')
        emb_size = 64

        self.heads = 8
        self.in_channels = emb_size
        self.out_channels = emb_size // self.heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = torch.nn.Linear(self.in_channels, self.heads * self.out_channels, bias=True)
        self.lin_r = torch.nn.Linear(self.in_channels, self.heads * self.out_channels, bias=True)

        self.att = torch.nn.Parameter(torch.Tensor(1, self.heads, self.out_channels * 3))

        # output_layer
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.lin_l.weight)
        init.orthogonal_(self.lin_r.weight)
        init.orthogonal_(self.att)

    def freeze_normalization(self):
        pass

    def reset_normalization(self):
        pass

    @property
    def frozen(self):
        return False


    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        H, C = self.heads, self.out_channels

        x_l = self.lin_l(left_features)
        x_r = self.lin_r(right_features)

        out = self.propagate(edge_indices, x=(x_l, x_r), size=(left_features.shape[0], right_features.shape[0]), edge_features=edge_features)
        return self.output_module(torch.cat([out, right_features], dim=-1))

    def message(self, x_j, x_i,
                index,
                edge_features):
        x = torch.cat([x_i, x_j, edge_features], dim=-1)
        x = torch.nn.functional.leaky_relu(x, self.negative_slope)
        x = x.view(-1, self.heads, self.out_channels * 3)
        alpha = (x * self.att).sum(dim=-1)
        alpha = torch_geometric.utils.softmax(alpha, index)
        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)
        x = x_j.view(-1, self.heads, self.out_channels) * alpha.unsqueeze(-1)
        return x.view(-1, self.heads * self.out_channels)