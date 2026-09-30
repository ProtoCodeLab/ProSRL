import math
import torch
import torch.optim as optim
import datetime
import itertools
from imblearn.over_sampling import RandomOverSampler
from TGAT_Test import  WGNN_test
from ParsingSource import *
from Tools import *
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import torch.nn as nn
import pandas as pd
import os
from model.gcn import AdversarialGAT
import torch.nn.functional as F
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.cluster import MiniBatchKMeans
from pwmmd_ex import PrototypeWeightedMMD,update_prototypes_epoch, update_target_prototypes_epoch



def generate_mask(g, critical_nodes):
    node_types = g.ndata['feat']
    critical_nodes_tensor = torch.tensor(critical_nodes, device=node_types.device)
    mask = torch.isin(node_types, critical_nodes_tensor).float()
    return mask.to(DEVICE)

# Initial super-parameter of network
init_gnn_params = {'EMBED_DIM': 32, 'N_HIDDEN_NODE': 100, 'N_EPOCH': 50, 'BATCH_SIZE': 32, 'LEARNING_RATE': 1e-3,
                   'MOMEMTUN': 0.9, 'L2_WEIGHT': 0.005, 'DROPOUT': 0.5, 'STRIDE': 1, 'PADDING': 0, 'POOL_SIZE': 2,'DICT_SIZE':0, 'TOKEN_SIZE': 0}

# Adjustable parameter
REGENERATE = False

dump_data_path = 'cdata_loader/'
if not os.path.exists(dump_data_path):
    os.makedirs(dump_data_path)

# Fixed parameter
IMBALANCE_PROCESSOR = RandomOverSampler()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
root_path_source = '/home/user/ProSRL/data/projects/'
root_path_csv = '/home/user/ProSRL/data/csvs/'
package_heads = ['org', 'gnu', 'bsh', 'javax', 'com']

# Start Time
start_time = datetime.datetime.now()
start_time_str = start_time.strftime('%Y-%m-%d_%H.%M.%S')

WGNN = {'acc': [], 'auc': [], 'f1': [], 'mcc': [], 'gmean': [], 'precision': [], 'recall': []}

# Get a list of source and target projects
path_train_and_test = []
with open('./data/pairs_CPDP.txt', 'r') as file_obj:
    for line in file_obj.readlines():
        line = line.strip('\n')
        line = line.strip(' ')
        path_train_and_test.append(line.split(','))


# ===============================
# Attention-based Multi-Prototype Helper
# ===============================
def attention_prototype_logits(features, prototypes, temperature=0.1):
    """
    features: [B,D]
    prototypes: [C,K,D]
    return: class_logits [B,C]
    """
    B,D = features.shape
    C,K,_ = prototypes.shape

    features = F.normalize(features, dim=1)
    prototypes = F.normalize(prototypes, dim=2)

    # Cosine similarity
    sims = torch.einsum('bd,ckd->bck', features, prototypes)  # [B,C,K]

    # Attention over K prototypes
    attn = F.softmax(sims / temperature, dim=2)  # [B,C,K]

    # Weighted sum to get class logits
    class_logits = (attn * sims).sum(dim=2)  # [B,C]
    return class_logits



for path in path_train_and_test[:2]:

    # Get file
    path_train_source = root_path_source + path[0]
    path_train_handcraft = root_path_csv + path[0] + '.csv'
    path_test_source = root_path_source + path[1]
    path_test_handcraft = root_path_csv + path[1] + '.csv'

    # Regenerate token or get from dump_data
    print(path[0] + "===" + path[1])
    source_target= path[0]+'_'+path[1]
    train_project_name = path[0]
    test_project_name = path[1]
    path_train_and_test_set = dump_data_path + train_project_name + '_to_' + test_project_name
    # If you don't need to regenerate, get it directly from dump_data
    if os.path.exists(path_train_and_test_set) and not REGENERATE:
        print("Loading saved data")
        obj = load_data(path_train_and_test_set)
        [train_graphs, test_graphs, train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size] = obj
    else:
        # Get a list of instances of the training and test sets
        print("Regenerate")
        train_file_instances = extract_handcraft_instances(path_train_handcraft)
        test_file_instances = extract_handcraft_instances(path_test_handcraft)

        # Get tokens
        dict_token_train = parse_source(path_train_source, train_file_instances, package_heads)
        dict_token_test = parse_source(path_test_source, test_file_instances, package_heads)

        # Turn tokens into numbers token
        list_dict, vector_len, vocabulary_size1,vocabulary1 = transform_token_to_number([dict_token_train, dict_token_test])

        vocabulary_size, vocabulary = load_data(os.path.join('/home/user/ProSRL/data_loader/dic/', 'global_vocabulary.pkl'))#/home/fdb/AI_Project/TLDP

        dict_encoding_train = list_dict[0]
        dict_encoding_test = list_dict[1]

        # Take out data that can be used for training
        train_ast, train_hand_craft, train_label = extract_data(path_train_handcraft, dict_encoding_train)
        test_ast, test_hand_craft, test_label = extract_data(path_test_handcraft, dict_encoding_test)

        # Imbalanced processing
        train_ast, train_hand_craft, train_label, balanced_file_instances = imbalance_process(train_ast, train_hand_craft, train_label, dict_token_train, IMBALANCE_PROCESSOR)

        print("Balanced file instances count:", len(balanced_file_instances))

        # Create graphs from AST with self-loops
        def create_graphs_from_ast(project_root_path, file_instances, package_heads, device, vocabulary=None):
            dict_graphs = extract_data_and_generate_graphs(project_root_path, file_instances, package_heads, graph_vocabulary = vocabulary)

            graphs = []
            missing_files = []

            for qualified_name, graph in dict_graphs:
                if graph is not None:
                    graph = graph.to(device)
                    graphs.append(graph)
                else:
                    missing_files.append(qualified_name)

            if missing_files:
                print("Files without graphs:", missing_files[:10])

            return graphs

        # Create graphs for training and testing data
        train_graphs = create_graphs_from_ast(path_train_source, balanced_file_instances, package_heads, DEVICE, vocabulary = vocabulary)

        test_graphs = create_graphs_from_ast(path_test_source, dict_token_test.keys(), package_heads, DEVICE, vocabulary = vocabulary)

        # Saved to dump_data
        obj = [train_graphs, test_graphs, train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len, vocabulary_size]
        dump_data(path_train_and_test_set, obj)

    train_src = load_data(os.path.join('/home/user/ProSRL/data_loader/autodl-tmp/', f'{train_project_name}_indices01.pkl'))
    test_tar = load_data(os.path.join('/home/user/ProSRL/data_loader/autodl-tmp/', f'{test_project_name}_indices01.pkl'))

    intersection_set = set(train_src) & set(test_tar)
    critical_nodes = list(intersection_set)
    class ASTDataset(DGLDataset):
        def __init__(self, graphs, ast, labels, device):
            self.graphs = [g.to(device) for g in graphs]
            self.ast = torch.tensor(ast).to(device)
            self.labels = torch.tensor(labels).to(device)
            super().__init__(name='ast_dataset')

        def process(self):
            pass

        def __getitem__(self, idx):
            return self.graphs[idx], self.ast[idx], self.labels[idx]

        def __len__(self):
            return len(self.graphs)

    # Define the dataset
    train_dataset = ASTDataset(train_graphs, train_ast, train_label,DEVICE)
    test_dataset = ASTDataset(test_graphs, test_ast, test_label,DEVICE)

    train_loader = GraphDataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = GraphDataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

    # Select nn parameters
    gnn_params = init_gnn_params.copy()
    gnn_params['DICT_SIZE'] = vocabulary_size + 1
    gnn_params['gcn_hidden']=64

    # Prototype Semantic Weighted MMD
    # =====================================================
    pswmmd = PrototypeWeightedMMD(
        gamma=5,
        temperature=0.1,
        defect_class_idx=1
    ).to(DEVICE)

    feat_dim = gnn_params['gcn_hidden']
    num_classes=2
    K_clean = 1
    K_defect = 5

    warmup_epochs=8
    lambda_max=0.3
    momentum=0.95

    src_prototypes = {
        0: torch.zeros(K_clean, feat_dim).to(DEVICE),
        1: torch.zeros(K_defect, feat_dim).to(DEVICE)
    }

    src_proto_initialized = {
        0: torch.zeros(K_clean,dtype=torch.bool).to(DEVICE),
        1: torch.zeros(K_defect, dtype=torch.bool).to(DEVICE)}

    Kt = K_clean + K_defect

    tgt_prototypes = torch.zeros(Kt,feat_dim).to(DEVICE)

    tgt_proto_initialized = torch.zeros(Kt,dtype=torch.bool).to(DEVICE)


    # ------------------ GNN training begins ------------------
    GNN_acc, GNN_auc, GNN_f1, GNN_mcc, GNN_gmean, GNN_precision, GNN_recall = [], [], [], [], [], [], []

    model = AdversarialGAT(gnn_params['gcn_hidden'],vocab_size = vocabulary_size + 1)
    model.to(DEVICE)

    best_defect_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0
    patience = 15
    lambda_mmd=6



    optimizer = optim.Adam(model.parameters(), lr=gnn_params['LEARNING_RATE'], betas=(0.9, 0.999), eps=1e-8, weight_decay=gnn_params['L2_WEIGHT'], amsgrad=False)

    acc_a, auc_a, f1_a, p_a, r_a, mcc_a, g_mean = [], [], [], [], [], [],[]
    for epoch in range(gnn_params['N_EPOCH']):
        total_defect_loss = 0
        total_loss_train = 0
        total_mp_loss = 0

        alpha = 0.5 * (1 + torch.cos(torch.tensor(epoch / gnn_params['N_EPOCH'] * 3.1416)))

        num_steps_per_epoch = len(train_loader)

        source_iter = iter(train_loader)
        target_iter = iter(test_loader)

        # ======= Epoch-level Prototype Update =======
        if epoch >= warmup_epochs:
            src_prototypes = update_prototypes_epoch(model=model,train_loader=train_loader,
                                                     prototypes=src_prototypes,
                                                     proto_initialized=src_proto_initialized,
                                                     K_clean=K_clean,K_defect=K_defect,momentum=momentum)

            tgt_prototypes = update_target_prototypes_epoch(model=model,target_loader=test_loader,
                                                            K=Kt, prototypes=tgt_prototypes,
                                                            proto_initialized=tgt_proto_initialized,
                                                            momentum=momentum)


        for step in range(num_steps_per_epoch):
            try:
                batch_graph, batch_ast_x, batch_y = next(source_iter)
            except StopIteration:
                source_iter = iter(train_loader)
                batch_graph, batch_ast_x, batch_y = next(source_iter)

            try:
                tgt_graph, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(test_loader)
                tgt_graph, _, _ = next(target_iter)

            batch_graph = batch_graph.to(DEVICE)
            batch_graph.ndata['feat'] = batch_graph.ndata['feat'].float().to(DEVICE)
            batch_ast_x = batch_ast_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            tgt_graph = tgt_graph.to(DEVICE)
            tgt_graph.ndata['feat'] = tgt_graph.ndata['feat'].float().to(DEVICE)
            tgt_domain_labels = torch.ones(tgt_graph.batch_size, dtype=torch.long).to(DEVICE)

            batched_graph = dgl.batch([batch_graph, tgt_graph]).to(DEVICE)
            mask = generate_mask(batched_graph, critical_nodes)

            model.train()

            _, defect_prob, x_src_mmd, x_tar_mmd, x_loss_mmd_node = model(batched_graph, batch_graph, tgt_graph, alpha, mask=None, compute_node_mmd=False)
            criterion = nn.BCELoss().to(DEVICE)

            src_defect_prob = defect_prob[:batch_graph.batch_size]
            tar_defect_prob = defect_prob[batch_graph.batch_size:]

            defect_loss = F.binary_cross_entropy(src_defect_prob, batch_y.float().view(-1, 1) )

            if epoch < warmup_epochs:

                loss_pswmmd = torch.tensor(
                    0.0,
                    device=DEVICE
                )

            else:

                loss_pswmmd = pswmmd(src_feat=x_src_mmd,tgt_feat=x_tar_mmd,
                    src_labels=batch_y.long(),
                    src_prototypes=src_prototypes,
                    tgt_prototypes=tgt_prototypes
                )

            # =============================================
            # Total Loss
            # =============================================
            if epoch < warmup_epochs:

                loss = defect_loss

            else:

                loss = (defect_loss + lambda_mmd* loss_pswmmd)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            total_mp_loss += loss_pswmmd.item()
            total_defect_loss += defect_loss.item()

        avg_defect_loss = total_defect_loss / num_steps_per_epoch
        avg_train_loss = total_loss_train / num_steps_per_epoch
        avg_mp_loss = total_mp_loss / num_steps_per_epoch



    GNN_acc, GNN_auc, GNN_f1, GNN_mcc, GNN_gmean, GNN_P, GNN_R = WGNN_test(critical_nodes, test_tar, model, train_graphs, test_graphs, train_label, test_label, GNN_acc, GNN_auc, GNN_f1, GNN_mcc, GNN_gmean, GNN_precision, GNN_recall)

    print(f"Average - F1: {GNN_f1[-1]:.3f}, Accuracy: {GNN_acc[-1]:.3f},"
          f" MCC: {GNN_mcc[-1]:.3f}, AUC: {GNN_auc[-1]:.3f},"
          f" G-Mean: {GNN_gmean[-1]:.3f}, Precision: {GNN_precision[-1]:.3f}, Recall: {GNN_recall[-1]:.3f}")



    WGNN['acc'].append(GNN_acc)
    WGNN['auc'].append(GNN_auc)
    WGNN['f1'].append(GNN_f1)
    WGNN['mcc'].append(GNN_mcc)
    WGNN['gmean'].append(GNN_gmean)
    WGNN['precision'].append(GNN_precision)
    WGNN['recall'].append(GNN_recall)

avg_acc = np.mean(WGNN['acc'])
avg_auc = np.mean(WGNN['auc'])
avg_f1 = np.mean(WGNN['f1'])
avg_mcc = np.mean(WGNN['mcc'])
avg_gmean = np.mean(WGNN['gmean'])
avg_precision = np.mean(WGNN['precision'])
avg_recall = np.mean(WGNN['recall'])

print(f"Average - F1: {avg_f1:.3f}, Accuracy: {avg_acc:.3f},"
      f" MCC: {avg_mcc:.3f}, AUC: {avg_auc:.3f},"
      f" G-Mean: {avg_gmean:.3f}, Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}")

row = {
    "F1": round(avg_f1, 3),
    "Accuracy": round(avg_acc, 3),
    "MCC": round(avg_mcc, 3),
    "AUC": round(avg_auc, 3),
    "G-Mean": round(avg_gmean, 3),
    "Precision": round(avg_precision, 3),
    "Recall": round(avg_recall, 3)
}

csv_path = "average_metrics_log.csv"
write_header = not os.path.exists(csv_path)
pd.DataFrame([row]).to_csv(csv_path, mode='a', index=False, header=write_header)

# End Time
end_time = datetime.datetime.now()
print(end_time - start_time)