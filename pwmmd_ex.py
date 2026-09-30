import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
import torch
import numpy as np
class PrototypeWeightedMMD(nn.Module):

    def __init__(
            self,
            gamma=5,
            temperature=0.1,
            defect_class_idx=1):

        super().__init__()

        self.gamma = gamma
        self.temperature = temperature
        self.defect_class_idx = defect_class_idx

    # =====================================================
    # RBF Kernel
    # =====================================================
    def _mix_rbf_kernel(self, X, Y):

        XX = torch.matmul(X, X.t())
        YY = torch.matmul(Y, Y.t())
        XY = torch.matmul(X, Y.t())

        X_sqnorms = torch.diag(XX)
        Y_sqnorms = torch.diag(YY)

        K_XX = torch.exp(
            -self.gamma *
            ( -2 * XX + X_sqnorms.unsqueeze(1) +X_sqnorms.unsqueeze(0)))

        K_YY = torch.exp( -self.gamma *(
                -2 * YY +Y_sqnorms.unsqueeze(1) +
                Y_sqnorms.unsqueeze(0)
            )
        )

        K_XY = torch.exp(
            -self.gamma *
            (
                -2 * XY +
                X_sqnorms.unsqueeze(1) +
                Y_sqnorms.unsqueeze(0)
            )
        )

        return K_XX, K_XY, K_YY

    # =====================================================
    # Prototype-guided Feature Attraction
    # =====================================================
    def feature_attraction_source(self,features, prototypes):
        features = F.normalize(features,dim=1)

        ###################################
        # flatten prototype
        ###################################

        proto = torch.cat([prototypes[0], prototypes[1]],dim=0)

        proto = F.normalize(proto,dim=1)

        ###################################
        # attention
        ###################################

        sim = torch.mm(features,proto.t())

        attn = F.softmax(sim / self.temperature,dim=1)

        semantic_proto = torch.mm(attn,proto)

        ###################################
        # adaptive attraction
        ###################################

        weights = self.source_semantic_weight(features,None,prototypes)

        eta = 0.2

        features_new = ((1 - eta) * features +eta * semantic_proto)

        features_new = F.normalize(features_new,dim=1)

        return features_new

    def feature_attraction_target(self,features,src_prototypes,tgt_prototypes):

        features = F.normalize(features,dim=1)

        ##########################################
        # source prototype
        ##########################################

        src_proto = torch.cat([src_prototypes[0],src_prototypes[1]],dim=0)

        src_proto = F.normalize(src_proto,dim=1)

        ##########################################
        # target prototype
        ##########################################

        tgt_proto = F.normalize(tgt_prototypes,dim=1)

        ##########################################
        # prototype transfer
        ##########################################

        proto_sim = torch.mm(tgt_proto,src_proto.t())

        proto_attn = F.softmax(proto_sim / self.temperature,dim=1)

        transferred_proto = torch.mm(proto_attn, src_proto)

        ##########################################
        # sample attention
        ##########################################

        sample_sim = torch.mm(features, transferred_proto.t())

        sample_attn = F.softmax(sample_sim / self.temperature,dim=1)

        semantic_feature = torch.mm(sample_attn,transferred_proto)

        ##########################################
        # adaptive attraction
        ##########################################

        weights = self.target_semantic_weight(features,src_prototypes,tgt_prototypes)

        eta =  weights.unsqueeze(1)

        features_new = ((1 - eta) * features + eta * semantic_feature)

        features_new = F.normalize(features_new,dim=1)

        return features_new

    # =====================================================
    # Prototype Attention
    # =====================================================
    def prototype_attention(self,features,prototypes):

        features = F.normalize(features,dim=1)

        prototypes = F.normalize(prototypes,dim=2)

        sim = torch.einsum('bd,ckd->bck',features,prototypes)
        attn = F.softmax(sim / self.temperature,dim=2)

        return attn

    # =====================================================
    # Source Prototype Semantic Relevance
    # =====================================================
    def source_semantic_weight(self,features,labels,prototypes):
        features = F.normalize(features, dim=1)

        clean_proto = F.normalize(prototypes[0],dim=1)

        defect_proto = F.normalize(prototypes[1],dim=1)

        #########################################
        # Clean prototype similarity
        #########################################

        clean_sim = torch.mm(features,clean_proto.t())

        clean_score = clean_sim.squeeze(1)

        #########################################
        # Defect prototype similarity
        #########################################

        defect_sim = torch.mm(features,defect_proto.t())

        defect_attn = F.softmax(defect_sim / self.temperature,dim=1)

        defect_score = (defect_attn *defect_sim).sum(dim=1)

        #########################################
        # Two semantic scores
        #########################################

        semantic = torch.stack([clean_score,defect_score],dim=1)

        semantic = F.softmax(semantic,dim=1)

        weights = semantic[:, 1]

        return weights

    # =====================================================
    # Target Prototype Semantic Transfer
    # =====================================================
    def target_semantic_weight(self,features,src_prototypes,tgt_prototypes):

        features = F.normalize(features,dim=1)

        ##########################################
        # flatten source prototype
        ##########################################

        src_proto = torch.cat([src_prototypes[0],src_prototypes[1]],dim=0)

        src_proto = F.normalize(src_proto,dim=1)

        tgt_proto = F.normalize(tgt_prototypes,dim=1)

        ##########################################
        # Prototype transfer
        ##########################################

        proto_sim = torch.mm(tgt_proto,src_proto.t())

        proto_attn = F.softmax(proto_sim / self.temperature,dim=1)

        semantic_bank = torch.cat([torch.zeros(src_prototypes[0].size(0),device=features.device),
            torch.ones(src_prototypes[1].size(0),device=features.device)])

        proto_semantic = torch.zeros(tgt_proto.size(0),2,device=features.device)

        for c in range(2):
            mask = semantic_bank == c

            proto_semantic[:, c] = (proto_attn[:, mask]).sum(dim=1)

        ##########################################
        # sample -> prototype
        ##########################################

        sample_sim = torch.mm(features,tgt_proto.t())
        sample_attn = F.softmax( sample_sim / self.temperature,dim=1)

        sample_semantic = torch.mm(sample_attn,proto_semantic)

        semantic = F.softmax(sample_semantic,dim=1)

        weights = semantic[:, 1]

        return weights

    # =====================================================
    # Weighted MMD
    # =====================================================
    def compute_weighted_mmd(self,Xs,Xt,ws,wt):
        ws = ws.clamp(min=0.05,max=0.95)
        wt = wt.clamp(min=0.05,max=0.95)

        ws = ws / ws.sum()
        wt = wt / wt.sum()


        ws = ws.view(-1, 1)
        wt = wt.view(-1, 1)

        ws = ws / (ws.sum() + 1e-8)
        wt = wt / (wt.sum() + 1e-8)

        Kss, Kst, Ktt = self._mix_rbf_kernel(Xs,Xt)
        Wss = torch.matmul(ws,ws.t())
        Wtt = torch.matmul(wt,wt.t())
        Wst = torch.matmul(ws,wt.t())

        loss_ss = (Wss * Kss).sum()
        loss_tt = (Wtt * Ktt).sum()
        loss_st = (Wst * Kst).sum()
        loss = (loss_ss+loss_tt-2.0 * loss_st)
        return loss

    # =====================================================
    # Forward
    # =====================================================
    def forward(self,src_feat,tgt_feat,src_labels,src_prototypes,tgt_prototypes):


        src_feat = self.feature_attraction_source(src_feat,src_prototypes)
        tgt_feat = self.feature_attraction_target(tgt_feat,src_prototypes,tgt_prototypes)


        src_weights = self.source_semantic_weight(src_feat,src_labels,src_prototypes)

        tgt_weights = self.target_semantic_weight(tgt_feat,src_prototypes,tgt_prototypes)

        mmd_loss = self.compute_weighted_mmd(src_feat,tgt_feat,src_weights,tgt_weights)

        return mmd_loss

# =====================================================
# Hungarian Matching
# =====================================================
def match_prototypes(old_proto, new_proto):

    old_norm = F.normalize(old_proto, dim=1)
    new_norm = F.normalize(new_proto, dim=1)

    sim = torch.mm(old_norm, new_norm.t())

    cost = (-sim).cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(cost)

    return col_ind

# =====================================================
# Source Prototype Update
# shape:
# [num_classes, K, feat_dim]
# =====================================================
def update_prototypes_epoch(model,train_loader,prototypes,
                            proto_initialized,K_clean=1,K_defect=3, momentum=0.95):

    model.eval()

    DEVICE = prototypes[0].device

    clean_features = []
    defect_features = []

    with torch.no_grad():

        for (batch_graph, batch_ast_x, batch_y) in train_loader:

            batch_graph = batch_graph.to(DEVICE)

            batch_graph.ndata['feat'] = (batch_graph.ndata['feat'].float().to(DEVICE))

            batch_y = (batch_y.view(-1).long().to(DEVICE))

            batched_graph = dgl.batch([batch_graph, batch_graph]).to(DEVICE)

            (_,_,x_src_mmd,_,_) = model(batched_graph,batch_graph,batch_graph,
                                        alpha=0,mask=None,compute_node_mmd=False)

            clean_mask = (batch_y == 0)

            defect_mask = (batch_y == 1)

            if clean_mask.sum() > 0:

                clean_features.append(
                    x_src_mmd[clean_mask])

            if defect_mask.sum() > 0:

                defect_features.append(
                    x_src_mmd[defect_mask])

    # ==========================
    # Clean Prototype
    # ==========================
    if len(clean_features) > 0:

        clean_feats = torch.cat( clean_features,dim=0)

        center = clean_feats.mean( dim=0,keepdim=True)

        if not proto_initialized[0].all():

            prototypes[0] = center

            proto_initialized[0][:] = True

        else:

            prototypes[0] = (momentum* prototypes[0]+(1-momentum)* center)

    # ==========================
    # Defect Prototype
    # ==========================
    if len(defect_features) > 0:

        defect_feats = torch.cat(defect_features,dim=0)

        defect_np = (defect_feats.cpu().numpy())

        if len(defect_np) >= K_defect:

            kmeans = MiniBatchKMeans(n_clusters=K_defect,batch_size=16,
                                     random_state=0,n_init=10)

            kmeans.fit(defect_np)

            centers = torch.tensor(kmeans.cluster_centers_,
                                   device=DEVICE,dtype=torch.float)

        else:

            mean_center = torch.tensor(defect_np.mean(axis=0),device=DEVICE,
                                       dtype=torch.float)

            centers = mean_center.unsqueeze(0).repeat(K_defect,1)

        if not proto_initialized[1].all():

            prototypes[1] = centers

            proto_initialized[1][:] = True

        else:

            match_idx = match_prototypes(prototypes[1],centers)

            centers = centers[match_idx]

            prototypes[1] = F.normalize(momentum* prototypes[1]+(1-momentum)* centers,dim=1)

    return prototypes


# =====================================================
# Target Prototype Update
#
# shape:
# [K, feat_dim]
#
# No Label
# =====================================================
def update_target_prototypes_epoch(model,target_loader,K=4,
                                   prototypes=None,proto_initialized=None,
                                   momentum=0.95):

    model.eval()

    DEVICE = prototypes.device

    all_features = []

    with torch.no_grad():

        for (tgt_graph, _,_) in target_loader:

            tgt_graph = tgt_graph.to(DEVICE)

            tgt_graph.ndata['feat'] = (tgt_graph.ndata['feat'].float().to(DEVICE))

            batched_graph = dgl.batch([tgt_graph, tgt_graph]).to(DEVICE)

            ( _,_,_,x_tar_mmd, _) = model(batched_graph,tgt_graph, tgt_graph,alpha=0,
                                          mask=None,compute_node_mmd=False)

            all_features.append(x_tar_mmd)

    if len(all_features) == 0:

        return prototypes

    feats = torch.cat(all_features,dim=0)

    feats_np = (feats.cpu().numpy())

    if len(feats_np) >= K:

        kmeans = MiniBatchKMeans(n_clusters=K,batch_size=16, random_state=0,n_init=10)

        kmeans.fit( feats_np)

        centers = torch.tensor(kmeans.cluster_centers_,device=DEVICE, dtype=torch.float)

    else:

        mean_center = torch.tensor(feats_np.mean(axis=0),device=DEVICE,dtype=torch.float)

        centers = mean_center.unsqueeze(0).repeat(K,1)

    if not proto_initialized.all():

        prototypes[:] = centers

        proto_initialized[:] = True

        return prototypes

    match_idx = match_prototypes(prototypes,centers)

    centers = centers[match_idx]

    prototypes[:] = F.normalize(momentum* prototypes+(1-momentum)* centers,dim=1)

    return prototypes