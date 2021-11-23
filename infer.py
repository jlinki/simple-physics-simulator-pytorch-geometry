# Imports
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing#, radius_graph
from torch_cluster import radius_graph

import reading_utils
import learnable_simulator

# Create directories for tensorboard
from torch.utils.tensorboard import SummaryWriter
os.makedirs('train_log', exist_ok=True)
os.makedirs('rollouts', exist_ok=True)

# Constants
INPUT_SEQUENCE_LENGTH = 6
batch_size = 2
noise_std = 6.7e-4
training_steps = int(2e7)
log_steps = 5
eval_steps = 20
save_steps = 100
model_path = "train_log/run0/model.pth" # 'model425000.pth' # Model path to trained model; Important when infering
data_path = 'data'
device = 'cuda'

# Load information from .json file
with open(data_path + '/metadata.json', 'rt') as f:
    metadata = json.loads(f.read())
num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
normalization_stats = {
    'acceleration': {
        'mean':torch.FloatTensor(metadata['acc_mean']).to(device), 
        'std':torch.sqrt(torch.FloatTensor(metadata['acc_std'])**2 + noise_std**2).to(device),
    }, 
    'velocity': {
        'mean':torch.FloatTensor(metadata['vel_mean']).to(device), 
        'std':torch.sqrt(torch.FloatTensor(metadata['vel_std'])**2 + noise_std**2).to(device),
    }, 
}

def eval_single_rollout(simulator, features, num_steps, device):
    initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:]
    
    current_positions = initial_positions
    predictions = []
    for step in range(num_steps):
        next_position = simulator.predict_positions(
            current_positions,
            n_particles_per_example=features['n_particles_per_example'],
            particle_types=features['particle_type'],
        ) # (n_nodes, 2)
        # Update kinematic particles from prescribed trajectory.
        kinematic_mask = (features['particle_type'] == 3).clone().detach().to(device)
        next_position_ground_truth = ground_truth_positions[:, step]
        kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, 2)
        next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)
        predictions.append(next_position)
        current_positions = torch.cat([current_positions[:, 1:], next_position[:, None, :]], dim=1)
    predictions = torch.stack(predictions) # (time, n_nodes, 2)
    ground_truth_positions = ground_truth_positions.permute(1,0,2)
    loss = (predictions - ground_truth_positions) ** 2
    output_dict = {
        'initial_positions': initial_positions.permute(1,0,2).cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'particle_types': features['particle_type'].cpu().numpy(),
    }
    return output_dict, loss

def eval_rollout(ds, simulator, num_steps, num_eval_steps=1, save_results=False, device='cuda'):
    eval_loss = []
    i = 0
    simulator.eval()
    with torch.no_grad():
        for example_i, (features, labels) in enumerate(ds):
            features['position'] = torch.tensor(features['position']).to(device) # (n_nodes, 600, 2)
            features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device)
            features['particle_type'] = torch.tensor(features['particle_type']).to(device)
            labels = torch.tensor(labels).to(device)
            example_rollout, loss = eval_single_rollout(simulator, features, num_steps, device)
            example_rollout['metadata'] = metadata
            eval_loss.append(loss)
            if save_results:
                example_rollout['metadata'] = metadata
                filename = f'rollout_{example_i}.pkl'
                filename = os.path.join('rollouts/', filename)
                with open(filename, 'wb') as f:
                    pickle.dump(example_rollout, f)
            i += 1
            if i >= num_eval_steps:
                break
    simulator.train()
    return torch.stack(eval_loss).mean(0)

def infer(simulator):
    ds = reading_utils.prepare_data_from_tfds(data_path=data_path+'/valid.tfrecord', is_rollout=True, metadata=metadata)
    eval_rollout(ds, simulator, num_steps=num_steps, save_results=True, device=device)

if __name__ == '__main__':
    simulator = learnable_simulator.Simulator(
        particle_dimension=2,
        node_in=30,
        edge_in=3,
        latent_dim=128,
        num_message_passing_steps=10,
        mlp_num_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata['default_connectivity_radius'],
        boundaries=np.array(metadata['bounds']),
        normalization_stats=normalization_stats,
        num_particle_types=9,
        particle_type_embedding_size=16,
        device=device,
    )
    if model_path is not None:
        simulator.load(model_path)
    if device == 'cuda':
        simulator.cuda()
    infer(simulator)