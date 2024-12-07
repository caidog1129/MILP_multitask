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

from MIPDataset_backdoor import GraphDataset as GraphDataset_backdoor
from MIPDataset_pas import GraphDataset as GraphDataset_pas
from GAT import GATPolicy_multitask as GATPolicy
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
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

def filter_data_file(sample_files):
    print("starting to filter bad data files")
    bad_list = ["MVC_barabasi_albert_297.lp","MVC_barabasi_albert_295.lp","MVC_barabasi_albert_287.lp","MVC_barabasi_albert_298.lp","MVC_barabasi_albert_28.lp","MVC_barabasi_albert_292.lp","MVC_barabasi_albert_280.lp"]+\
        ["CA_2000_4000_519.lp"]
    corrupted_files = 0
    valid_files = []
    has_iLB = 0
    for i,file in enumerate(sample_files):
        BGFilepath, solFilePath = file
        corrupted = False
        for bad_file in bad_list:
            if bad_file in solFilePath:
                corrupted = True
        # with open(BGFilepath, "rb") as f:
        #     try:
        #         bgData = pickle.load(f)
        #     except:
        #         corrupted = True
        with open(solFilePath, "rb") as f:
            try:
                solData = pickle.load(f)
            except:
                corrupted = True
        if corrupted:
            corrupted_files += 1
        else:
            if "neg_examples_iLB_0" in solData:
                has_iLB += 1
                #print(file)
                valid_files.append(file)
            else:
                print(file,"has no iLB")
        if i%50==0: print("processed %d files"%(i+1))
            
    print("filted out %d corrupted files"%(corrupted_files), ";", has_iLB, "has iLB")
    #exit()
    return valid_files

parser = argparse.ArgumentParser()
parser.add_argument('--taskName', type=str, default="IP")
parser.add_argument('--pretrainModel', type=str, default=None)
parser.add_argument('--temp',type=float, default=0.07)
parser.add_argument('--perturb',type=float, default=0.05)
parser.add_argument('--fracdata',type=float, default=1)
parser.add_argument('--weight', type=bool, default=False)
parser.add_argument('--freeze', type=int, default = 0)
parser.add_argument('--negex', type=str,default=None) # can be "iLB" or "perturb"
parser.add_argument('--instance_dir1', type=str)
parser.add_argument('--instance_dir2', type=str)
parser.add_argument('--result_dir1', type=str)
parser.add_argument('--result_dir2', type=str)

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
LEARNING_RATE1 = 0.00001
LEARNING_RATE2 = 0.00001
NB_EPOCHS =9999
BATCH_SIZE = 16
NUM_WORKERS = 0
WEIGHT_NORM=100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infoNCE_loss_function = losses.NTXentLoss(temperature=args.temp, distance=DotProductSimilarity()).to(DEVICE)

# 1st task
sample_names = os.listdir(args.result_dir1)
sample_files = [(os.path.join(args.instance_dir1,name), os.path.join(args.result_dir1,name)) for name in sample_names]
sample_files = sorted(sample_files)
    
random.seed(0)
random.shuffle(sample_files)

sample_files = sample_files[:int(len(sample_files)*args.fracdata)]

train_files = sample_files[int(0.2 * len(sample_files)):]
valid_files = sample_files[:int(0.20 * len(sample_files))]
print("Training on", int(0.80 * len(sample_files)), "instances")
print("Validating on", int(0.2 * len(sample_files)), "instances")

train_data = GraphDataset_backdoor(train_files)
train_loader1 = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, follow_batch=['x_vars'])
valid_data = GraphDataset_backdoor(valid_files)
valid_loader1 = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, follow_batch=['x_vars'])


# 2nd task
sample_names = os.listdir(args.result_dir2)
sample_files = [(os.path.join(args.instance_dir2,name).replace('.sol',''), os.path.join(args.result_dir2,name)) for name in sample_names]
sample_files = sorted(sample_files)
sample_files = filter_data_file(sample_files)
    
random.seed(67)
random.shuffle(sample_files)

sample_files = sample_files[:int(len(sample_files)*args.fracdata)][:200]

train_files = sample_files[int(0.2 * len(sample_files)):]
valid_files = sample_files[:int(0.20 * len(sample_files))]
print("Training on", int(0.80 * len(sample_files)), "instances")
print("Validating on", int(0.2 * len(sample_files)), "instances")

train_data = GraphDataset_pas(train_files, args)
train_loader2 = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_data = GraphDataset_pas(valid_files, args)
valid_loader2 = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

PredictModel = GATPolicy().to(DEVICE)

# if not args.pretrainModel is None:
#     state = torch.load(args.pretrainModel, map_location=DEVICE)
#     PredictModel.load_state_dict(state)
#     print(f"loaded model from {args.pretrainModel}")
for name, param in PredictModel.named_parameters():
    if 'conv_v_to_c' not in name and 'conv_c_to_v' not in name and 'embedding' not in name:
        param.requires_grad = False

def train(epoch, predict, data_loader1, data_loader2, optimizer1=None, optimizer2=None, weight_norm=1):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """

    if optimizer1 and optimizer2:
        predict.train()
    else:
        predict.eval()
    mean_loss1 = 0
    mean_loss2 = 0
    n_samples_processed1 = 0
    n_samples_processed2 = 0

    iterator1 = iter(data_loader1)
    iterator2 = iter(data_loader2)

    while True:
        try:
            # Load a batch from both task-specific datasets
            batch1 = next(iterator1).to(DEVICE)
            batch2 = next(iterator2).to(DEVICE)
        except StopIteration:
            # If any iterator is exhausted, break the loop
            break

        with torch.set_grad_enabled(optimizer1 is not None):
            batch1 = batch1.to(DEVICE)
            output = predict(batch1.x_cons,
                        batch1.edge_index_cons_to_vars,
                        batch1.edge_attr,
                        batch1.x_vars, 1)

            pred, max_length = pad_by_batch(output, batch1.x_vars_batch)
            embeddings = torch.sigmoid(pred)
            anchor_positive = []
            anchor_negative = []
            positive_idx = []
            negative_idx = []
            total_sample = len(batch1)

            for i in range(len(batch1)):
                for j in range(len(batch1.pos_sample[i])):
                    anchor_positive.append(i)
                    positive_idx.append(total_sample)
                    tensor = torch.tensor(batch1.pos_sample[i][j])
                    padding_size = max_length - len(tensor)
                    pad = torch.full((padding_size,), 0, device=tensor.device, dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, pad]).unsqueeze(0)
                    embeddings = torch.cat([embeddings, padded_tensor.to(DEVICE)])
                    total_sample += 1
                for j in range(len(batch1.neg_sample[i])):
                    anchor_negative.append(i)
                    negative_idx.append(total_sample)
                    tensor = torch.tensor(batch1.neg_sample[i][j])
                    padding_size = max_length - len(tensor)
                    pad = torch.full((padding_size,), 0, device=tensor.device, dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, pad]).unsqueeze(0)
                    embeddings = torch.cat([embeddings, padded_tensor.to(DEVICE)])
                    total_sample += 1

            triplets = (torch.tensor(anchor_positive).to(DEVICE), torch.tensor(positive_idx).to(DEVICE), torch.tensor(anchor_negative).to(DEVICE), torch.tensor(negative_idx).to(DEVICE))
            loss1 = infoNCE_loss_function(embeddings, indices_tuple = triplets)
            if optimizer1 is not None:
                writer.add_scalar('train_loss1', loss1, epoch)
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
            else:
                writer.add_scalar('valid_loss1', loss1, epoch)
            mean_loss1 += loss1
            n_samples_processed1 += len(batch1)
            
        with torch.set_grad_enabled(optimizer2 is not None):
            batch2 = batch2.to(DEVICE)
            # get target solutions in list format
            solInd = batch2.nsols
            negInd = batch2.nnegsamples
            target_sols = []
            target_vals = []
            target_negs = []
            instances = []
            solEndInd = 0
            valEndInd = 0
            negSampEndInd = 0

            for i in range(solInd.shape[0]):#for in batch
                nvar = len(batch2.varInds[i][0][0])
                solStartInd = solEndInd
                solEndInd = solInd[i] * nvar + solStartInd
                valStartInd = valEndInd
                valEndInd = valEndInd + solInd[i]
                
                sols = batch2.solutions[solStartInd:solEndInd].reshape(-1, nvar)
                vals = batch2.objVals[valStartInd:valEndInd]

                target_sols.append(sols)
                target_vals.append(vals)
                
                negSampStartInd = negSampEndInd 
                negSampEndInd = negInd[i] * nvar + negSampStartInd
                negs = batch2.negsamples[negSampStartInd:negSampEndInd].reshape(-1, nvar)
                target_negs.append(negs)

            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            batch2.constraint_features[torch.isinf(batch2.constraint_features)] = 10 #remove nan value
            #predict the binary distribution, BD
            BD = predict(
                batch2.constraint_features,
                batch2.edge_index,
                batch2.edge_attr,
                batch2.variable_features, 
                2
            )
            BD = BD.sigmoid()
    
            # calculate weights
            index_arrow = 0
            embeddings = None

            total_samples = 0
            anchor_positive = []
            positive_idx = []
            anchor_negative = []
            negative_idx = []
            
            for ind,(sols,vals,negs) in enumerate(zip(target_sols,target_vals, target_negs)):       
                # get a binary mask
                varInds = batch2.varInds[ind]
                varname_map=varInds[0][0]
                b_vars=varInds[1][0].long()

                #get binary variables
                sols = sols[:,varname_map][:,b_vars]

                # cross-entropy
                n_var = batch2.ntvars[ind]
                pre_sols = BD[index_arrow:index_arrow + n_var].squeeze()[b_vars]

                negs = negs[:,varname_map][:,b_vars]

                cur_embeddings = torch.cat([pre_sols.reshape(1,-1), sols, negs])

                anchor_positive = anchor_positive + [total_samples] * sols.shape[0]
                anchor_negative = anchor_negative + [total_samples] * negs.shape[0]
                positive_idx = positive_idx + list(range(total_samples + 1, total_samples + 1 + sols.shape[0]))
                negative_idx = negative_idx + list(range(total_samples + 1 + sols.shape[0], total_samples + 1 + sols.shape[0] + negs.shape[0]))
                total_samples += 1 + sols.shape[0] + negs.shape[0]

                if embeddings is None:
                    embeddings = cur_embeddings
                else:
                    embeddings = torch.cat([embeddings,cur_embeddings])
            
            triplets = (torch.tensor(anchor_positive).to(DEVICE), torch.tensor(positive_idx).to(DEVICE), torch.tensor(anchor_negative).to(DEVICE), torch.tensor(negative_idx).to(DEVICE))
            loss2 = infoNCE_loss_function(embeddings, indices_tuple = triplets)

            if optimizer2 is not None:
                writer.add_scalar('train_loss2', loss2, epoch)
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()
            else:
                writer.add_scalar('valid_loss2', loss2, epoch)
            mean_loss2 += loss2
            n_samples_processed2 += len(batch2)

    mean_loss1 = mean_loss1 / n_samples_processed1
    mean_loss2 = mean_loss2 / n_samples_processed2
    mean_loss = mean_loss1 / 3 + mean_loss2
    return mean_loss, mean_loss1, mean_loss2

optimizer1 = torch.optim.Adam(PredictModel.parameters(), lr=LEARNING_RATE1)
optimizer2 = torch.optim.Adam(PredictModel.parameters(), lr=LEARNING_RATE2)
best_val_loss = 99999

if not args.pretrainModel is None:
    valid_loss,_,_ = train(epoch, PredictModel, valid_loader1, valid_loader2, None)
    print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}")
    best_val_loss = valid_loss

# if args.freeze == 1:
#     print("freezing some layers")
#     for param in PredictModel.parameters():
#         param.requires_grad = False
#     for param in PredictModel.output_module.parameters():
#         param.requires_grad = True

for epoch in range(NB_EPOCHS):
    print(f"Start epoch {epoch}")
    begin=time.time()
    train_loss,train_loss1,train_loss2 = train(epoch, PredictModel, train_loader1, train_loader2, optimizer1, optimizer2)
    print(f"Epoch {epoch} Train loss: {train_loss:0.3f}, Train loss1: {train_loss1:0.3f}, Train loss2: {train_loss2:0.3f}")
    valid_loss,valid_loss1,valid_loss2 = train(epoch, PredictModel, valid_loader1, valid_loader2, None, None)
    print(f"Epoch {epoch} Valid loss: {valid_loss:0.3f}, Valid loss1: {valid_loss1:0.3f}, Valid loss2: {valid_loss2:0.3f}")
        
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
