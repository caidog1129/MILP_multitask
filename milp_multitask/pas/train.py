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
from GAT import GATPolicy
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
#this file is to train a predict model. given a instance's bipartite graph as input, the model predict the binary distribution.

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

def EnergyWeightNorm(task):
    if task=="IP":
        return 1
    elif task=="WA":
        return 100
    elif task=="WA_90_1500":
        return 100
    elif task == "IS":
        return -100
    elif task == "CA":
        return -1000
    elif task == "CA_2000_4000":
        return -10000
    elif task == "CA_3000_6000":
        return -10000
    elif task == "CA_400_2000":
        return -1000
    elif task == "INDSET_BA4_3000":
        return -200
    elif task == "INDSET_BA5_6000":
        return -200
    elif task == "MVC_BA5_3000":
        return 100
    elif task == "MVC_BA5_6000":
        return 200

parser = argparse.ArgumentParser()
parser.add_argument('--taskName', type=str, default="IP")
parser.add_argument('--pretrainModel', type=str, default=None)
parser.add_argument('--temp',type=float, default=0.07)
parser.add_argument('--perturb',type=float, default=0.05)
parser.add_argument('--fracdata',type=float, default=1)
parser.add_argument('--weight', type=bool, default=False)
parser.add_argument('--freeze', type=int, default = 0)
parser.add_argument('--negex', type=str,default="iLB") # can be "iLB" or "perturb"
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
LEARNING_RATE = 0.00002
NB_EPOCHS =9999
BATCH_SIZE = 16
NUM_WORKERS = 0
WEIGHT_NORM=100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

infoNCE_loss_function = losses.NTXentLoss(temperature=args.temp, distance=DotProductSimilarity()).to(DEVICE)

sample_names = os.listdir(args.result_dir)
sample_files = [(os.path.join(args.instance_dir,name).replace('.sol',''), os.path.join(args.result_dir,name)) for name in sample_names]
sample_files = sorted(sample_files)
sample_files = filter_data_file(sample_files)
    
random.seed(67)
random.shuffle(sample_files)

sample_files = sample_files[:int(len(sample_files)*args.fracdata)][:200]

train_files = sample_files[int(0.2 * len(sample_files)):]
valid_files = sample_files[:int(0.20 * len(sample_files))]
print("Training on", int(0.80 * len(sample_files)), "instances")
print("Validating on", int(0.2 * len(sample_files)), "instances")

train_data = GraphDataset(train_files, args)
train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_data = GraphDataset(valid_files, args)
valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

PredictModel = GATPolicy().to(DEVICE)

if not args.pretrainModel is None:
    PredictModel.load_state_dict(torch.load(args.pretrainModel), strict=False)
    print(f"loaded model from {args.pretrainModel}")
    if args.freeze == 1:
        for name, param in PredictModel.named_parameters():
            if 'conv_v_to_c' in name or 'conv_c_to_v' in name or 'embedding' in name:
                param.requires_grad = False

def train(epoch, predict, data_loader, optimizer=None,weight_norm=1):
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
            # get target solutions in list format
            solInd = batch.nsols
            negInd = batch.nnegsamples
            target_sols = []
            target_vals = []
            target_negs = []
            instances = []
            solEndInd = 0
            valEndInd = 0
            negSampEndInd = 0

            for i in range(solInd.shape[0]):#for in batch
                nvar = len(batch.varInds[i][0][0])
                solStartInd = solEndInd
                solEndInd = solInd[i] * nvar + solStartInd
                valStartInd = valEndInd
                valEndInd = valEndInd + solInd[i]
                
                sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
                vals = batch.objVals[valStartInd:valEndInd]

                target_sols.append(sols)
                target_vals.append(vals)
                
                negSampStartInd = negSampEndInd 
                negSampEndInd = negInd[i] * nvar + negSampStartInd
                negs = batch.negsamples[negSampStartInd:negSampEndInd].reshape(-1, nvar)
                target_negs.append(negs)

            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10 #remove nan value
            #predict the binary distribution, BD
            BD = predict(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
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
                varInds = batch.varInds[ind]
                varname_map=varInds[0][0]
                b_vars=varInds[1][0].long()

                #get binary variables
                sols = sols[:,varname_map][:,b_vars]

                # cross-entropy
                n_var = batch.ntvars[ind]
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
