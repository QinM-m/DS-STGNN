import torch
from torch import nn

from torch_scatter import scatter_mean, scatter_max, scatter_add

# from mot_neural_solver.models.mlp import MLP
from models.mlp import MLP


class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self, edge_model=None, node_model=None):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
        """
        super(MetaLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)

        Returns: Updated Node and Edge Feature matrices

        """
        row, col = edge_index

        # Edge Update
        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr)

        # Node Update
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr)

        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)

class EdgeModel(nn.Module):
    """
    Class used to peform the edge update during Neural message passing
    """
    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """
    def __init__(self, node_mlp, node_agg_fn):
        super(NodeModel, self).__init__()

        self.node_mlp = node_mlp
        self.node_agg_fn = node_agg_fn

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        # flow_out_mask = row < col
        # flow_out_row, flow_out_col = row[flow_out_mask], col[flow_out_mask]
        # flow_out_input = torch.cat([x[flow_out_col], edge_attr[flow_out_mask]], dim=1)
        # flow_out = self.flow_out_mlp(flow_out_input)
        # flow_out = self.node_agg_fn(flow_out, flow_out_row, x.size(0))
        #
        # flow_in_mask = row > col
        # flow_in_row, flow_in_col = row[flow_in_mask], col[flow_in_mask]
        # flow_in_input = torch.cat([x[flow_in_col], edge_attr[flow_in_mask]], dim=1)
        # flow_in = self.flow_in_mlp(flow_in_input)
        #
        # flow_in = self.node_agg_fn(flow_in, flow_in_row, x.size(0))
        # flow = torch.cat((flow_in, flow_out), dim=1)

        flow = torch.cat([x[row], edge_attr], dim=1)
        flow_updated = self.node_mlp(flow)
        agg_messages_nodes = self.node_agg_fn(flow_updated, row, x.size(0))

        return agg_messages_nodes

class MLPGraphIndependent(nn.Module):
    """
    Class used to to encode (resp. classify) features before (resp. after) neural message passing.
    It consists of two MLPs, one for nodes and one for edges, and they are applied independently to node and edge
    features, respectively.

    This class is based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, edge_in_dim = None, node_in_dim = None, edge_out_dim = None, node_out_dim = None,
                 node_fc_dims = None, edge_fc_dims = None, dropout_p = None, use_batchnorm = None):
        super(MLPGraphIndependent, self).__init__()

        if node_in_dim is not None :
            self.node_mlp = MLP(input_dim=node_in_dim, fc_dims=list(node_fc_dims) + [node_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.node_mlp = None

        if edge_in_dim is not None :
            self.edge_mlp = MLP(input_dim=edge_in_dim, fc_dims=list(edge_fc_dims) + [edge_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.edge_mlp = None

    def forward(self, edge_feats = None, nodes_feats = None):

        if self.node_mlp is not None and nodes_feats is not None:
            out_node_feats = self.node_mlp(nodes_feats)

        else:
            out_node_feats = nodes_feats

        if self.edge_mlp is not None and edge_feats is not None:
            out_edge_feats = self.edge_mlp(edge_feats)

        else:
            out_edge_feats = edge_feats

        return out_edge_feats, out_node_feats

class MOTMPNet(nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - 2 encoder MLPs (1 for nodes, 1 for edges) that provide the initial node and edge embeddings, respectively,
    - 4 update MLPs (3 for nodes, 1 per edges used in the 'core' Message Passing Network
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output.

    This class was initially based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """

    def __init__(self, model_params, bb_encoder = None, arch=None):
        """
        Defines all components of the model
        Args:
            bb_encoder: (might be 'None') CNN used to encode bounding box apperance information.
            model_params: dictionary contaning all model hyperparameters
        """
        super(MOTMPNet, self).__init__()

        self.node_cnn = bb_encoder
        self.model_params = model_params

        # Define Encoder and Classifier Networks
        edges_params = model_params['encoder_feats_dict']['edges']
        nodes_params = model_params['encoder_feats_dict']['nodes'][arch]
        edges_params.update(nodes_params)
        encoder_feats_dict = edges_params
        classifier_feats_dict = model_params['classifier_feats_dict']

        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        self.classifier = MLPGraphIndependent(**classifier_feats_dict)

        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.spatial_MPNet, self.temporal_MPNet = self._build_core_MPNet(model_params=model_params, encoder_feats_dict=encoder_feats_dict)

        self.num_enc_steps = model_params['num_enc_steps']
        self.num_class_steps = model_params['num_class_steps']

        node_out_dim = encoder_feats_dict['node_out_dim']


        node_out_dim = encoder_feats_dict['node_out_dim']


        self.temporal_proj = nn.Sequential(
            nn.Linear(node_out_dim, node_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_out_dim, node_out_dim)
        )


        self.fusion_gate = nn.Sequential(
            nn.Linear(node_out_dim * 2, node_out_dim),
            nn.Sigmoid()
        )


        self.mpn_fusion_gate = nn.Sequential(
            nn.Linear(node_out_dim * 2, node_out_dim),
            nn.Sigmoid()
        )

        nn.init.zeros_(self.temporal_proj[-1].weight)
        nn.init.zeros_(self.temporal_proj[-1].bias)

    def _build_core_MPNet(self, model_params, encoder_feats_dict):
        node_agg_fn = model_params['node_agg_fn']
        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)
        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]
        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        self.reattach_initial_nodes = model_params['reattach_initial_nodes']
        self.reattach_initial_edges = model_params['reattach_initial_edges']

        edge_factor = 2 if self.reattach_initial_edges else 1
        node_factor = 2 if self.reattach_initial_nodes else 1

        edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * encoder_feats_dict[
            'edge_out_dim']
        node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim']

        edge_model_feats_dict = model_params['edge_model_feats_dict']
        node_model_feats_dict = model_params['node_model_feats_dict']


        spatial_edge_mlp = MLP(input_dim=edge_model_in_dim, fc_dims=edge_model_feats_dict['fc_dims'],
                               dropout_p=edge_model_feats_dict['dropout_p'],
                               use_batchnorm=edge_model_feats_dict['use_batchnorm'])
        spatial_node_mlp = MLP(input_dim=node_model_in_dim, fc_dims=node_model_feats_dict['fc_dims'],
                               dropout_p=node_model_feats_dict['dropout_p'],
                               use_batchnorm=node_model_feats_dict['use_batchnorm'])

        spatial_MPNet = MetaLayer(edge_model=EdgeModel(edge_mlp=spatial_edge_mlp),
                                  node_model=NodeModel(node_mlp=spatial_node_mlp, node_agg_fn=node_agg_fn))


        temporal_node_mlp = MLP(input_dim=node_model_in_dim, fc_dims=node_model_feats_dict['fc_dims'],
                                dropout_p=node_model_feats_dict['dropout_p'],
                                use_batchnorm=node_model_feats_dict['use_batchnorm'])

        temporal_MPNet = MetaLayer(edge_model=None,
                                   node_model=NodeModel(node_mlp=temporal_node_mlp, node_agg_fn=node_agg_fn))

        return spatial_MPNet, temporal_MPNet

    def forward(self, data):

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr


        has_temporal = hasattr(data, 'prev_x') and hasattr(data,
                                                           'temporal_edge_index') and data.temporal_edge_index.numel() > 0


        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x)
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats


        if has_temporal:
            _, prev_latent_node_feats = self.encoder(data.prev_edge_attr, data.prev_x)
            prev_proj = self.temporal_proj(prev_latent_node_feats)


            aligned_prev_proj = torch.zeros_like(latent_node_feats)


            prev_idx = data.temporal_edge_index[0].clamp(max=prev_proj.size(0) - 1)
            t_idx = data.temporal_edge_index[1].clamp(max=latent_node_feats.size(0) - 1)


            aligned_prev_proj[t_idx] = prev_proj[prev_idx]


            gate = self.fusion_gate(torch.cat([latent_node_feats, aligned_prev_proj], dim=-1))
            latent_node_feats = gate * latent_node_feats + (1 - gate) * aligned_prev_proj
        # =====================================================================

        first_class_step = self.num_enc_steps - self.num_class_steps + 1
        outputs_dict = {'classified_edges': []}

        for step in range(1, self.num_enc_steps + 1):
            if self.reattach_initial_edges:
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes:
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)


            latent_node_feats_s, latent_edge_feats = self.spatial_MPNet(latent_node_feats, edge_index,
                                                                        latent_edge_feats)


            if has_temporal:

                in_dim = latent_node_feats.size(1) + latent_edge_feats.size(1)
                temporal_in = torch.zeros(latent_node_feats.size(0), in_dim, device=latent_node_feats.device)


                repeat_factor = 2 if self.reattach_initial_nodes else 1
                node_dim = aligned_prev_proj.size(1)
                temporal_in[:, :node_dim * repeat_factor] = aligned_prev_proj.repeat(1, repeat_factor)


                latent_node_feats_t = self.temporal_MPNet.node_model.node_mlp(temporal_in)


                fusion_context = torch.cat([latent_node_feats_s, latent_node_feats_t], dim=-1)


                fusion_weight = self.mpn_fusion_gate(fusion_context)

                latent_node_feats = latent_node_feats_s + fusion_weight * latent_node_feats_t
            else:
                latent_node_feats = latent_node_feats_s

            if step >= first_class_step:
                dec_edge_feats, _ = self.classifier(latent_edge_feats)
                outputs_dict['classified_edges'].append(dec_edge_feats)

        if self.num_enc_steps == 0:
            dec_edge_feats, _ = self.classifier(latent_edge_feats)
            outputs_dict['classified_edges'].append(dec_edge_feats)

        return outputs_dict