import torch
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torch.nn as nn
from math import sqrt
import random
import pickle as pkl
from refile import smart_load_image, smart_open, smart_listdir
import pdb


def smart_pkl_load(file_name):
    if file_name.startswith("s3://"):
        with smart_open(file_name, "rb") as f:
            features = pkl.load(f)
    else:
        with open(file_name, "rb") as f:
            features = pkl.load(f)
    return features


def get_text_emb_from_file(text_feat_path, cap_path, text_type, num_prompts, text_proxy_factor, one_cap_per_image=False):
    """
    Get all text embedding from a directory
    :param dir_name: directory name contains text features
    :return: tensor, (51, 512)
    """
    print("loading text feat for training")
    text_feat = torch.from_numpy(smart_pkl_load(text_feat_path))
    caps = smart_pkl_load(cap_path)
    print("loaded text feat ({}) for training and corresponding captions ({})".format(text_feat.shape, len(caps)))
    if text_type == "cls":
        text_feat = rearrange(text_feat, 'b t f -> (b t) f')
    # else:
    #     N = text_feat.shape[0]
    #     # if text_proxy_factor is a percentage
    #     if text_proxy_factor <= 1:
    #         k = int(N * text_proxy_factor)
    #     # if text_proxy_factor is a number
    #     else:
    #         text_proxy_factor = min(text_proxy_factor, N)
    #         k = text_proxy_factor
    #     text_feat = text_feat[torch.randperm(N)[:k]]

    return text_feat


def triplet_loss(u, v):
    """
    Triplet loss
    :param u: (N, emb_size)
    :param v: (N, 50, emb_size)
    :return: triplet loss value
    """
    from pytorch_metric_learning import losses
    # (N, 50, emb_size)
    loss_func = losses.TripletMarginLoss()

    u_tiled = u.unsqueeze(1).repeat(1, v.shape[1], 1)
    u_flatten = rearrange(u_tiled, 'b p f -> (b p) f')
    v_flatten = rearrange(v, 'b p f -> (b p) f')
    zeros = torch.zeros(len(v_flatten), )
    ones = torch.ones(len(u_flatten), )
    labels = torch.cat([zeros, ones])
    embeddings = torch.cat([v_flatten, u_flatten])
    loss = loss_func(embeddings, labels)
    return loss


class TextMatchingLoss(nn.Module):
    """
    Text based matching loss
    """
    def __init__(self, text_feat_path, text_cap_path, text_type, num_prompts, text_proxy_ratio, metric_func="MSE"):
        super(TextMatchingLoss, self).__init__()
        super().__init__()
        self.text_type = text_type
        self.text_proxy_ratio = text_proxy_ratio
        self.num_prompts = num_prompts
        self.text_emb = get_text_emb_from_file(text_feat_path, text_cap_path, text_type, num_prompts, text_proxy_ratio, one_cap_per_image=False)
        if metric_func == "MSE":
            self.distance_metric = nn.MSELoss()
        elif metric_func == "l1":
            self.distance_metric = nn.L1Loss()
        elif metric_func == "js":
            self.distance_metric = JSD()
        else:
            print("invalid metric function")

    def forward(self, patch_emb, cmag_emb):
        """
        f = 512
        :param self.text_emb: (t, f) mean @ 80 prompts, t = 259 for shapenet51_10k_small
        :param patch_emb: (b, p, f), p = 166 for patch_cc@10
        :param cmag_emb: (b, f)
        :return: loss in float
        """
        b = patch_emb.shape[0]
        device = patch_emb.get_device()
        device = "cpu" if device == -1 else device

        N = self.text_emb.shape[0]
        # if text_proxy_factor is a percentage
        if self.text_proxy_ratio <= 1:
            k = int(N * self.text_proxy_factor)
        # if text_proxy_factor is a number
        else:
            text_proxy_ratio = min(self.text_proxy_ratio, N)
            k = text_proxy_ratio

        if self.text_type == "caption":
            text_feat = self.text_emb[torch.randperm(N)[:k]]
        else:
            text_feat = self.text_emb

        text_feat = text_feat.to(device)
        patch_emb_reshape = rearrange(patch_emb, 'b p f -> (b p) f')
        normalized_patch_emb = patch_emb_reshape / patch_emb_reshape.norm(dim=-1, keepdim=True)
        normalized_text_emb = text_feat / text_feat.norm(dim=-1, keepdim=True)
        normalized_cmag_emb = cmag_emb / cmag_emb.norm(dim=-1, keepdim=True)

        query_proxy_feat = normalized_text_emb

        print("proxy feat shape: {}".format(query_proxy_feat.shape))
        print("patch cc feat shape: {}".format(normalized_patch_emb.shape))
        print("cmag feat shape: {}".format(normalized_cmag_emb.shape))

        k_p_similarity, k_c_similarity = cal_matching_loss(query_proxy_feat, normalized_patch_emb, normalized_cmag_emb, b)

        loss = self.distance_metric(k_p_similarity, k_c_similarity) * 100.0
        return loss


class PatchMatchingLoss(nn.Module):
    """
    Patch based matching loss
    """
    def __init__(self, cc_num, matching_loss_type="all", metric_func="MSE"):
        super(PatchMatchingLoss, self).__init__()
        if metric_func == "MSE":
            self.distance_metric = nn.MSELoss()
        elif metric_func == "l1":
            self.distance_metric = nn.L1Loss()
        elif metric_func == "js":
            self.distance_metric = JSD()
        else:
            print("invalid metric function")
        self.matching_loss_type = matching_loss_type
        layer_dict = {
            1: [1],
            2: [1, 4],
            3: [1, 4, 9],
            4: [1, 4, 9, 16],
            5: [1, 4, 9, 25],
            6: [1, 4, 9, 16, 36],
            7: [1, 4, 9, 16, 49],
            8: [1, 4, 9, 25, 64],
            9: [1, 4, 9, 16, 25, 81],
            10: [1, 4, 9, 16, 36, 100],
            11: [1, 4, 9, 16, 36, 121],
            12: [1, 4, 9, 16, 25, 49, 144],
            13: [1, 4, 9, 16, 25, 49, 169],
            14: [1, 4, 9, 16, 25, 64, 196],
            15: [1, 4, 9, 16, 36, 64, 225],
        }
        self.layer_selection = layer_dict[cc_num]

    def forward(self, patch_emb, cmag_emb):
        """
        f = 512
        :param self.text_emb: (t, f) mean@80 prompt, t=51 for shapenet51_10k_small
        :param patch_emb: (b, p, f) p=166 for patch_cc@10
        :param cmag_emb: (b, f)
        :return: float
        """
        b, p = patch_emb.shape[0], patch_emb.shape[1]
        device = patch_emb.get_device()
        device = "cpu" if device == -1 else device
        # (b, p, f)
        normalized_patch_emb = patch_emb / patch_emb.norm(dim=-1, keepdim=True)
        # (b * p, f)
        normalized_patch_emb_reshape = rearrange(normalized_patch_emb, 'b p f -> (b p) f')
        # (b, f)
        normalized_cmag_emb = cmag_emb / cmag_emb.norm(dim=-1, keepdim=True)
        if self.matching_loss_type == "all":
            mask = (torch.ones(b * p) - torch.eye(b * p)).to(device)
            query_proxy_feat = normalized_patch_emb_reshape
        elif self.matching_loss_type == "layer_cascade":
            select_patch_emb = select_cascade_repr(normalized_patch_emb, self.layer_selection, method="layer_random")
            mask = None
            # q = 26 for cc@10
            query_proxy_feat = rearrange(select_patch_emb, 'b q f -> (b q) f')

        k_p_similarity, k_c_similarity = cal_matching_loss(query_proxy_feat, normalized_patch_emb_reshape, normalized_cmag_emb, b, mask=mask)

        loss = self.distance_metric(k_p_similarity, k_c_similarity) * 100.0
        return loss


# Additive Margin Softmax was proposed by paper(https://arxiv.org/abs/1801.05599)
class AMSoftmaxLoss(nn.Module):
    """
    AM-Softmax loss
    """
    def __init__(self, feature_dim, output_dim, margin, scaling_factor, feature_normalized=False):
        super(AMSoftmaxLoss, self).__init__()
        super().__init__()
        self.featurue_dim = feature_dim
        self.output_dim = output_dim
        self.margin = margin
        self.scaling_factor = scaling_factor
        self.feature_normalized = feature_normalized
        self.w = nn.Parameter(torch.randn(feature_dim, output_dim))

    def forward(self, embedding, labels):
        device = embedding.get_device()
        if device == -1:
            device = "cpu"
        w = self.w.to(device)
        x = embedding
        assert embedding.shape[0] == labels.shape[0]
        labels = labels.unsqueeze(1)
        labels_repeat = labels.repeat(1, x.shape[1])
        x = rearrange(x, 'b p f -> (b p) f')
        labels = rearrange(labels_repeat, 'b p -> (b p)')

        # normalize w
        w = w / (w ** 2).sum(0, keepdim=True) ** 0.5
        if not self.feature_normalized:
            # if the feature is not normalized, we'll normalize it for you
            norm = x.norm(dim=1, p=2, keepdim=True)
            x = x.div(norm.expand_as(x))

        x = torch.matmul(x, w)
        mask = labels.new_tensor(torch.arange(self.output_dim), device=device).reshape(1, -1) == labels.reshape(-1, 1)
        
        x_am = x - self.margin * mask.type(torch.float32)
        outputs = F.log_softmax(self.scaling_factor * x_am, dim=1)
        loss = F.nll_loss(outputs, labels)
        return loss


class JSD(nn.Module):
    """
    JS Divergence
    """

    def __init__(self):
        super(JSD, self).__init__()

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)

        m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), m, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), m, reduction="batchmean")
        return 0.5 * loss


def cal_matching_loss(query_proxy_feat, patch_feat, cmag_feat, b, mask=None):
    """
    Calculate matching loss
    :param query_proxy_feat: (k, f)
    :param patch_feat: (b * p, f)
    :param cmag_feat: (b, f)
    :return: query-proxy similarity matrix, value-proxy similarity matrix
    """""
    # (k, 512) x (512, b * p) -> (k, b * p)
    q_p_similarity = query_proxy_feat @ patch_feat.T

    # mask exists when patch matching loss is used.
    if mask is not None:
        q_p_similarity = q_p_similarity * mask

    # (k, b * p) -> (k, b, p)
    q_p_similarity = rearrange(q_p_similarity, 'k (b p) -> k b p', b=b)
    # (k, b, p) -> (k, b)
    q_p_similarity_max = q_p_similarity.max(-1)[0]

    # (k, 512) x (512, b) -> (k, b)
    k_c_similarity = query_proxy_feat @ cmag_feat.T

    # k_p_similarity_max[k_p_similarity_max>=0.9999] = 0.

    return q_p_similarity_max, k_c_similarity


def select_cascade_repr(patch_emb, layer_patch_num_list, method="layer_random"):
    """
    Randomly select patch-cc features from each layer
    :param patch_emb: (b, p, f)
    :param layer_patch_num_list: [1, 4, 9, 16, 36, 100]
    :param method: layer_random or uniform
    :return: selected patch
    """
    random.seed(0)
    if method == "layer_random":
        batch_patch_emb = []
        for data in patch_emb:
            selected_patch_list = []
            i = 0
            # [1, 2, 3, 4, 6, 10]
            for layer_patch_num in layer_patch_num_list:
                num_selected = int(sqrt(layer_patch_num))
                start_idx = i
                end_idx = i + layer_patch_num
                items = range(start_idx, end_idx)
                new_items = random.sample(items, k=num_selected)
                selected_patch = data[new_items, :]
                selected_patch_list.append(selected_patch)
                i = end_idx
            single_selected_patch_emb = torch.cat(selected_patch_list)
            batch_patch_emb.append(single_selected_patch_emb)
        b_patch_emb = torch.stack(batch_patch_emb)
        return b_patch_emb
