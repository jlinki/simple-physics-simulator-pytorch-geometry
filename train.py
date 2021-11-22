# Imports
import os
import json
import numpy as np
import torch


import learnable_simulator
import reading_utils

# Create directories for tensorboard
from tensorboardX import SummaryWriter
os.makedirs('train_log', exist_ok=True)
os.makedirs('rollouts', exist_ok=True)

# Constants
INPUT_SEQUENCE_LENGTH = 6
batch_size = 2 
noise_std = 6.7e-4
training_steps = int(5e5)
log_steps = 5
eval_steps = 20
save_steps = 100
model_path = None
data_path = 'data/Water'
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

def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]

def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence = time_diff(position_sequence)
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape)) * (noise_std_last_step/num_velocities**0.5)

    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

    return position_sequence_noise

#def _read_metadata(data_path):
#    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
#        return json.loads(fp.read())


def train(simulator):
    i = 0
    while os.path.isdir('train_log/run'+str(i)):
        i += 1
    LOG_DIR = 'train_log/run'+str(i)+'/'

    writer = SummaryWriter(LOG_DIR)

    lr_start = 1e-4
    lr_final = 1e-6
    lr_decay = 0.1
    lr_decay_steps = int(training_steps/4) #int(5e6)
    lr_new = lr_start
    optimizer = torch.optim.Adam(simulator.parameters(), lr=lr_start)

    dataset = reading_utils.prepare_data_from_tfds(data_path=data_path+'/train.tfrecord', batch_size=batch_size, metadata=metadata)

    step = 0
    try:
        for features, labels in dataset:   #todo the tensorflow warning appears here (4 times)
            features['position'] = torch.tensor(features['position']).to(device) #[424,6,2] todo: 6 = current + 5 past time step position
            features['n_particles_per_example'] = torch.tensor(features['n_particles_per_example']).to(device) # [2], [212,212] equals  212*torch.ones(2,dtype=int).to(device)
            features['particle_type'] = torch.tensor(features['particle_type']).to(device) #[424]
            labels = torch.tensor(labels).to(device) #[424,2]

            sampled_noise = get_random_walk_noise_for_position_sequence(features['position'], noise_std_last_step=noise_std).to(device)
            non_kinematic_mask = (features['particle_type'] != 3).clone().detach().to(device)
            sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

            pred, target = simulator.predict_accelerations(
                next_position=labels, 
                position_sequence_noise=sampled_noise, 
                position_sequence=features['position'], 
                n_particles_per_example=features['n_particles_per_example'], 
                particle_types=features['particle_type'],
            )
            loss = (pred - target) ** 2
            loss = loss.sum(dim=-1)
            num_non_kinematic = non_kinematic_mask.sum()

            loss = torch.where(non_kinematic_mask.bool(), loss, torch.zeros_like(loss)) #loss if condition, zeros if not condition -> only moving particles are updated
            loss = loss.sum() / num_non_kinematic

            if step % log_steps == 0:
                writer.add_scalar("training_loss", loss, step)
                writer.add_scalar("learing rate", lr_new, step)

                #with open(LOG_DIR + 'log.txt', 'a') as f:
                #    print("Training step: ", step, "Loss: ", loss.item(), file=f)
                #print("Training step: ", step,"/",training_steps, "Loss: ", loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if step % eval_steps == 0:
            #     eval_loss = eval_rollout(ds_eval, simulator, num_steps, num_eval_steps=10, device=device)
            #     writer.add_scalar("eval_loss", eval_loss, step)

            lr_new = lr_final + (lr_start - lr_final) * lr_decay ** (step/lr_decay_steps)
            #lr_new = lr_start * (lr_decay ** (step/lr_decay_steps))
            for g in optimizer.param_groups:
                g['lr'] = lr_new

            step += 1
            print(f'Training step: {step}/{training_steps}. Loss: {loss}.', end="\r",)
            if step >= training_steps:
                break

            if step % save_steps == 0:
                simulator.save(LOG_DIR+'model.pth')

    except KeyboardInterrupt:
        pass

    simulator.save(LOG_DIR+'model.pth')
    writer.close()


if __name__ == '__main__':
    simulator = learnable_simulator.Simulator(
        particle_dimension=2,
        node_in=30,
        edge_in=3,
        latent_dim=64,
        num_message_passing_steps=10,
        mlp_num_layers=1,
        mlp_hidden_dim=64,
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
    train(simulator)