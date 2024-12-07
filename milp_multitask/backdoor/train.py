import os
import torch
import torch_geometric
import random
import time
import argparse
import copy
from IPython import embed
import pickle
from pytorch_metric_learning import losses
from torchmetrics.functional import auroc
#from torchmetrics.classification import BinaryAccuracy
from torcheval.metrics import BinaryAccuracy
from pytorch_metric_learning.distances import DotProductSimilarity
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from MIPDataset import GraphDataset
# from MIPDataset import GraphDataset_filter as GraphDataset
from GAT import GATPolicy
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
#this file is to train a predict model. given a instance's bipartite graph as input, the model predict the binary distribution.
def pad_by_batch(tensor, x_vars_batch, value=0):
    """
    Organizes and pads the tensor based on batch indices given in x_vars_batch.

    Parameters:
    - tensor: The input tensor containing values.
    - x_vars_batch: A tensor containing batch indices for each value in tensor.
    - value: The value used for padding.

    Returns:
    - A 2D tensor with each row being a padded sequence.
    """
    unique_batches = torch.unique(x_vars_batch)
    grouped_tensors = [tensor[x_vars_batch == batch_idx] for batch_idx in unique_batches]

    max_length = max([len(t) for t in grouped_tensors])

    padded_tensors = []
    for t in grouped_tensors:
        padding_size = max_length - len(t)
        pad = torch.full((padding_size,), value, device=tensor.device, dtype=tensor.dtype)
        padded_tensor = torch.cat([t, pad])
        padded_tensors.append(padded_tensor)
        
    return torch.stack(padded_tensors), max_length

parser = argparse.ArgumentParser()
parser.add_argument('--taskName', type=str, default="IP")
parser.add_argument('--pretrainModel', type=str, default=None)
parser.add_argument('--temp',type=float, default=0.07)
parser.add_argument('--perturb',type=float, default=0.05)
parser.add_argument('--fracdata',type=float, default=1)
parser.add_argument('--weight', type=bool, default=False)
parser.add_argument('--freeze', type=int, default = 0)
parser.add_argument('--negex', type=str,default=None) # can be "iLB" or "perturb"
parser.add_argument('--instance_dir', type=str)
parser.add_argument('--result_dir', type=str)

args = parser.parse_args()

#set folder
train_task=args.taskName
if not os.path.isdir(f'./train_logs'):
    os.mkdir(f'./train_logs')
if not os.path.isdir(f'./train_logs/{train_task}'):
    os.mkdir(f'./train_logs/{train_task}')
if not os.path.isdir(f'./pretrain'):
    os.mkdir(f'./pretrain')
if not os.path.isdir(f'./pretrain/{train_task}'):
    os.mkdir(f'./pretrain/{train_task}')
model_save_path = f'./pretrain/{train_task}/'
log_save_path = f"train_logs/{train_task}/"
log_file = open(f'{log_save_path}{train_task}_train.log', 'wb')
writer = SummaryWriter(f'{log_save_path}')

#set params
LEARNING_RATE = 0.00001
NB_EPOCHS =9999
BATCH_SIZE = 16
NUM_WORKERS = 0
WEIGHT_NORM=100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infoNCE_loss_function = losses.NTXentLoss(temperature=args.temp, distance=DotProductSimilarity()).to(DEVICE)

sample_names = os.listdir(args.result_dir)
sample_files = [(os.path.join(args.instance_dir,name), os.path.join(args.result_dir,name)) for name in sample_names]
sample_files = sorted(sample_files)

#filter out bad files
# filter_sample_files = []
# for index in range(len(sample_files)):
#     insPath, resPath = sample_files[index]
#     df = pd.read_csv(resPath, index_col=0)
#     if df.iloc[0]["run_time"] <= 1 and df.iloc[-1]["run_time"] >= 1:
#         filter_sample_files.append(sample_files[index])
# sample_files = filter_sample_files        
        
random.seed(0)
random.shuffle(sample_files)

sample_files = sample_files[:int(len(sample_files)*args.fracdata)]

train_files = sample_files[int(0.2 * len(sample_files)):]
valid_files = sample_files[:int(0.20 * len(sample_files))]
print("Training on", int(0.80 * len(sample_files)), "instances")
print("Validating on", int(0.2 * len(sample_files)), "instances")

train_data = GraphDataset(train_files)
train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, follow_batch=['x_vars'])
valid_data = GraphDataset(valid_files)
valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, follow_batch=['x_vars'])

PredictModel = GATPolicy().to(DEVICE)

if not args.pretrainModel is None:
    PredictModel.load_state_dict(torch.load(args.pretrainModel), strict=False)
    print(f"loaded model from {args.pretrainModel}")
    if args.freeze == 1:
        for name, param in PredictModel.named_parameters():
            if 'conv_v_to_c' in name or 'conv_c_to_v' in name or 'embedding' in name:
                param.requires_grad = False

def train(epoch, predict, data_loader, optimizer=None, weight_norm=1):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """

    if optimizer:
        predict.train()
    else:
        predict.eval()
    mean_loss = 0
    n_samples_processed = 0
    
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            
            batch = batch.to(DEVICE)
            output = predict(batch.x_cons,
                        batch.edge_index_cons_to_vars,
                        batch.edge_attr,
                        batch.x_vars)

            pred, max_length = pad_by_batch(output, batch.x_vars_batch)
            embeddings = torch.sigmoid(pred)
            anchor_positive = []
            anchor_negative = []
            positive_idx = []
            negative_idx = []
            total_sample = len(batch)

            for i in range(len(batch)):
                for j in range(len(batch.pos_sample[i])):
                    anchor_positive.append(i)
                    positive_idx.append(total_sample)
                    tensor = torch.tensor(batch.pos_sample[i][j])
                    padding_size = max_length - len(tensor)
                    pad = torch.full((padding_size,), 0, device=tensor.device, dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, pad]).unsqueeze(0)
                    embeddings = torch.cat([embeddings, padded_tensor.to(DEVICE)])
                    total_sample += 1
                for j in range(len(batch.neg_sample[i])):
                    anchor_negative.append(i)
                    negative_idx.append(total_sample)
                    tensor = torch.tensor(batch.neg_sample[i][j])
                    padding_size = max_length - len(tensor)
                    pad = torch.full((padding_size,), 0, device=tensor.device, dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, pad]).unsqueeze(0)
                    embeddings = torch.cat([embeddings, padded_tensor.to(DEVICE)])
                    total_sample += 1

            triplets = (torch.tensor(anchor_positive).to(DEVICE), torch.tensor(positive_idx).to(DEVICE), torch.tensor(anchor_negative).to(DEVICE), torch.tensor(negative_idx).to(DEVICE))
            loss = infoNCE_loss_function(embeddings, indices_tuple = triplets)
            if optimizer is not None:
                writer.add_scalar('train_loss', loss, epoch)
            else:
                writer.add_scalar('valid_loss', loss, epoch)


            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += loss
            n_samples_processed += len(batch)
    mean_loss /= n_samples_processed
    return mean_loss

optimizer = torch.optim.Adam(PredictModel.parameters(), lr=LEARNING_RATE)
best_val_loss = 99999

# if not args.pretrainModel is None:
#     valid_loss = train(epoch, PredictModel, valid_loader, None)
#     print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}")
#     best_val_loss = valid_loss

    
# if args.freeze == 1:
#     print("freezing some layers")
#     for param in PredictModel.parameters():
#         param.requires_grad = False
#     for param in PredictModel.output_module.parameters():
#         param.requires_grad = True

for epoch in range(NB_EPOCHS):
    print(f"Start epoch {epoch}")
    begin=time.time()
    train_loss = train(epoch, PredictModel, train_loader, optimizer)
    print(f"Epoch {epoch} Train loss: {train_loss:0.3f}")
    valid_loss = train(epoch, PredictModel, valid_loader, None)
    print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}")
        
    if valid_loss<best_val_loss:
        best_val_loss = valid_loss
        torch.save(PredictModel.state_dict(),model_save_path+'model_best.pth')
    torch.save(PredictModel.state_dict(), model_save_path+'model_last.pth')
    st = f'@epoch{epoch}   Train loss:{train_loss}  Valid loss:{valid_loss} TIME:{time.time()-begin}\n'
    print(st,"\n===================")
    log_file.write(st.encode())
    log_file.flush()
writer.close()
print('done')
