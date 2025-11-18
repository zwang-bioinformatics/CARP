###################################

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, GPSConv, GINEConv, TopKPooling
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import scatter_mean

###################################

class CARP(torch.nn.Module):

    ###################################

    def resolve_layer(self, kind, params, n_dim = None, e_dim = None): 

        if kind == "TransformerConv": 
            return TransformerConv(
                in_channels=n_dim, 
                edge_dim=e_dim, 
                **params
            )

    ###################################

    def __init__(self, node_dim, edge_dim, config):

        ###################################

        super(CARP, self).__init__()

        self.config = config
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        ###################################

        node_in = node_dim
        edge_in = edge_dim

        ###################################

        self.top_k_ratio = config["top_k_ratio"]**(1/(
                int(config["gnn_model_config"]["depth"] > 0) + 
                int(config["gnn_interface_config"]["depth"] > 0) + 
                int(config["gps_config"] is not None)
            )
        )

        ###################################

        self.gps = None
        self.gps_topk = None

        if config["gps_config"] is not None: 
            
            conv = None

            if config["gps_config"]["conv"] == "GINEConv": 
                conv = GINEConv(
                    nn.Linear(node_in, node_in), 
                    edge_dim = edge_in
                )

            elif config["gps_config"]["conv"] == "TransformerConv": 
                conv = TransformerConv(
                    in_channels=node_in, 
                    out_channels = node_in, 
                    heads = 1, dropout = 0.0, 
                    edge_dim=edge_in
                )

            self.gps = GPSConv(
                channels = node_in, heads = 1,
                norm = GraphNorm(node_in),
                dropout = config["gps_config"]["dropout"],
                conv = conv
            )
            
            self.gps_topk = TopKPooling(node_in, ratio = self.top_k_ratio)

        ###################################
        
        self.gnn_model = nn.ModuleList()
        self.model_topk = None

        if config["gnn_model_config"]["depth"] > 0: 

            for _ in range(config["gnn_model_config"]["depth"]):
                self.gnn_model.append(
                    self.resolve_layer(
                        config["gnn_model_config"]["kind"], config["gnn_model_config"]["params"],
                        n_dim = node_in, e_dim = edge_in
                    )
                )
                node_in = config["gnn_model_config"]["params"]["out_channels"]
                self.gnn_model.append(GraphNorm(node_in))
                self.gnn_model.append(nn.ReLU())
            
            self.model_topk = TopKPooling(node_in, ratio = self.top_k_ratio)

        ###################################

        self.gnn_interface = nn.ModuleList()
        self.interface_topk = None

        if config["gnn_interface_config"]["depth"] > 0: 

            for _ in range(config["gnn_interface_config"]["depth"]):
                self.gnn_interface.append(
                    self.resolve_layer(
                        config["gnn_interface_config"]["kind"], config["gnn_interface_config"]["params"],
                        n_dim = node_in, e_dim = edge_in
                    )
                )
                node_in = config["gnn_interface_config"]["params"]["out_channels"]
                self.gnn_interface.append(GraphNorm(node_in))
                self.gnn_interface.append(nn.ReLU())
            
            self.interface_topk = TopKPooling(node_in, ratio = self.top_k_ratio)

        ###################################

        self.interface_pool = GlobalAttention(
            nn.Sequential(
                nn.Linear(node_in + 2, 1),
                nn.Dropout(p=config["interface_pool"]["dropout_gate"]),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(node_in + 2, config["interface_pool"]["out_channels"]), 
                nn.Dropout(p=config["interface_pool"]["dropout_map"]),
                nn.ReLU()
            )
        )

        self.interface_predictor = nn.Linear(config["interface_pool"]["out_channels"], 2)

        ###################################

        self.global_fold_pool = GlobalAttention(
            nn.Sequential(
                nn.Linear(node_in + 2, 1),
                nn.Dropout(p=config["global_fold_pool"]["dropout_gate"]),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(node_in + 2, config["global_fold_pool"]["out_channels"]), 
                nn.Dropout(p=config["global_fold_pool"]["dropout_map"]),
                nn.ReLU()
            )
        )

        self.global_fold_predictor = nn.Linear(config["global_fold_pool"]["out_channels"], 3)

        ###################################

        self.global_interface_pool = GlobalAttention(
            nn.Sequential(
                nn.Linear(node_in + 2, 1),
                nn.Dropout(p=config["global_interface_pool"]["dropout_gate"]),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Linear(node_in + 2, config["global_interface_pool"]["out_channels"]), 
                nn.Dropout(p=config["global_interface_pool"]["dropout_map"]),
                nn.ReLU()
            )
        )

        self.global_interface_predictor = nn.Linear(config["global_interface_pool"]["out_channels"], 3)

        ###################################

    def forward(self, nf, ef, ei, ic, pk, pri, tki, batch):

        ###################################

        # nf: node features
        # ef: edge features
        # ei: edge index
        # ic: interface connect
        # pk: polymer kind
        # pri: protein-rna interface
        # tki: top-k index
        # batch: batch index for each node 

        ###################################

        ni = batch.new_full((nf.shape[0], ), -1)

        ###################################

        if self.gps is not None: 
            # print("GPS batch:",batch.shape, batch.max(), batch.min())
            nf = self.gps(nf, ei, edge_attr = ef, batch = batch)
            nf = nn.functional.relu(nf)

        ###################################

        if self.gps_topk is not None: 

            nf, ei, ef, tki, perm,_ = self.gps_topk(
                x = nf, batch = tki,
                edge_index=ei, edge_attr=ef
            )

            batch = batch[perm]
            pk, pri, = pk[perm], pri[perm]

            if len(ic.shape) == 2: 
                ni[perm] = torch.arange(nf.shape[0], device=nf.device)
                ic[:,0] = ni[ic[:,0]]
                ic = ic[ic[:,0] >= 0]
                ni = ni[perm]

        ###################################

        for i in range(self.config["gnn_model_config"]["depth"]):

            if self.config["gnn_model_config"]["kind"] in ["TransformerConv"]: nf = self.gnn_model[3*i](nf, ei, ef)
            else: assert False, "NOT IMPLEMENTED!" 

            nf = self.gnn_model[3*i + 1](nf)
            nf = self.gnn_model[3*i + 2](nf)

        ###################################

        if self.model_topk is not None: 

            ni[:] = -1

            nf, ei, ef, tki, perm,_ = self.model_topk(
                x = nf, batch = tki,
                edge_index=ei, edge_attr=ef,
            )

            batch = batch[perm]
            pk, pri, = pk[perm], pri[perm]
            
            if len(ic.shape) == 2: 
                ni[perm] = torch.arange(nf.shape[0], device=nf.device)          
                ic[:,0] = ni[ic[:,0]]
                ic = ic[ic[:,0] >= 0]
                ni = ni[perm]
        
        ###################################

        if self.config["gnn_interface_config"]["depth"] > 0:

            em = torch.squeeze(torch.logical_xor(pk[ei[0]], pk[ei[1]]))
            ef = ef[em]
            ei = ei[:,em]
            del em

            for i in range(self.config["gnn_interface_config"]["depth"]):

                if self.config["gnn_interface_config"]["kind"] in ["TransformerConv"]: nf = self.gnn_interface[3*i](nf, ei, ef)
                else: assert False, "NOT IMPLEMENTED!" 

                nf = self.gnn_interface[3*i + 1](nf)
                nf = self.gnn_interface[3*i + 2](nf)

            ni[:] = -1

            nf, ei, ef, tki, perm, _ = self.interface_topk(
                x = nf, batch = tki,
                edge_index=ei, edge_attr=ef,
            )
            
            batch = batch[perm]
            pk, pri, = pk[perm], pri[perm]
            
            if len(ic.shape) == 2: 
                ni[perm] = torch.arange(nf.shape[0], device=nf.device)          
                ic[:,0] = ni[ic[:,0]]
                ic = ic[ic[:,0] >= 0]
                ni = ni[perm]

        ###################################

        nf = torch.cat((nf, pk, pri), dim=1)

        ###################################
        
        interface_out = None 

        if len(ic.shape) == 2: 
            interface_out = self.interface_pool(nf[ic[:,0]], ic[:,1])
            interface_out = self.interface_predictor(interface_out)

        ###################################

        global_interface_out = self.global_interface_pool(nf, batch)
        global_interface_out = self.global_interface_predictor(global_interface_out)

        ###################################

        global_fold_out = self.global_fold_pool(nf, batch)
        global_fold_out = self.global_fold_predictor(global_fold_out)

        ###################################

        return interface_out, torch.cat((global_fold_out, global_interface_out), dim=1)

###################################
