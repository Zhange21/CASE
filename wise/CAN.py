import copy
import random

import torch
from torch.nn import functional as F
from .utils import parent_module, brackets_to_periods, EarlyStopMeter, EditingMeanAct
import transformers
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from .merge import slerp, GTA, linear
import torch.nn as nn
import gc

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform
from tqdm import tqdm

# from python_tsp.heuristics import solve_tsp
import elkai
from sklearn.preprocessing import LabelEncoder
import math

merge_dict = {
    'slerp': slerp(),
    'ties': GTA('magnitude', 'sum', normalize=True),
    'magnitude_norm': GTA('magnitude', None, normalize=True),
    'magnitude': GTA('magnitude', None, normalize=False),
    'sign': GTA(None, 'sum', normalize=True),
    'dare_ties': GTA('rescaled_random', 'sum'),
    'dare_linear': GTA('random', None),
    'linear': linear()
}

edit_history = []
merge_group_edit_history = []

def euc(query, key, config, act_mask=None, infer=False):
    # Euclidean distance

    act_fn = ACT2FN[config.hidden_act]
    l2_norm = torch.norm(act_fn(key) - act_fn(query), dim=-1)
    if infer and l2_norm.size(1) > 100:
        topk = torch.topk(l2_norm, k=1, largest=True)
        return topk.values.mean()

    if act_mask is not None:
        return torch.sum(l2_norm * act_mask.to(l2_norm.device), dim=1) / torch.sum(act_mask.to(l2_norm.device), dim=1)
    else:
        return torch.mean(l2_norm, dim=-1)


class CAN(torch.nn.Module):
    def __init__(self, config, model, device):
        super(CAN, self).__init__()
        self.config = config
        self.model = model
        self.config = config
        if hasattr(self.model.config, 'hidden_act'):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            self.config.hidden_act = self.model.config.activation_function
        # self.tokenizer = model.tokenizer
        layer = config.inner_params[0]
        self.device = device
        self.adapter_layer = None
        self.original_layer = None

        # --- ensure proper formatting (WISE edits weights matrices) ---
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        # --- Add WISE to chosen layers ---
        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        self.layer_name = self.layer.rsplit(".", 1)[-1]
        adapter_layer = getattr(self.edit_module, self.layer_name)

        if type(adapter_layer) is not WISEAdapter:
            setattr(self.edit_module, self.layer_name, WISEAdapter(config, adapter_layer, transpose=transpose))
            self.original_layer = copy.deepcopy(adapter_layer)
            print(f"New weights successfully inserted into {layer}")

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Forward
    def __call__(self, **kwargs):

        if self.config.merge_moe and len(self.get_adapter_layer().moe_weight) > self.config.max_merge_num:
            if hasattr(self.get_adapter_layer(), 'editing') and not self.get_adapter_layer().editing:
                # final merge for moe
                if len(self.get_adapter_layer().moe_weight) > 0 and self.get_adapter_layer().editing_total_cnt >= self.config.merge_moe_freq:
                    print('length of memory is ', len(self.get_adapter_layer().moe_weight), '!!!!!!')
                    self.get_adapter_layer().merge_moe_weight_pair()
        return self.model(**kwargs)

    def reset_layer(self):
        layer = getattr(self.edit_module, self.layer_name)
        del layer
        setattr(self.edit_module, self.layer_name, self.get_adapter_layer().original_layer)

    def get_adapter_layer(self):
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is WISEAdapter, print('Adapter Layer is not added correctly....')
        return adapter_layer

    # TODO: generation
    def generate(self, *args, **kwargs):
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def sort_samples_merge(self, config, tokens_list, act_masks=None, deact_masks=None):

        def cosine_similarity(grad1, grad2):
            """计算两个梯度向量之间的余弦相似度"""
            grad1 = grad1.flatten()
            grad2 = grad2.flatten()
            return F.cosine_similarity(grad1.unsqueeze(0), grad2.unsqueeze(0), dim=-1).item()

        def merge_samples(sample_list):
            """合并相似的样本"""
            # 你可以根据需求来合并样本，这里提供了一个示例，直接对文本进行拼接。
            merged_sample = sample_list[0]  # 假设第一个样本作为合并的基准
            for sample in sample_list[1:]:
                # 示例操作：简单拼接文本，你可以按需更改
                merged_sample['input_ids'] = torch.cat([merged_sample['input_ids'], sample['input_ids']], dim=-1)
                merged_sample['attention_mask'] = torch.cat([merged_sample['attention_mask'], sample['attention_mask']], dim=-1)
                merged_sample['labels'] = torch.cat([merged_sample['labels'], sample['labels']], dim=-1)
            return merged_sample

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        loss_sample = []
        gradient_list = []
        # ori_samples = []
        all_grad = []

        for i in range(len(tokens_list)):
            self.model.zero_grad()

            # sample_i = {}
            # for keyi, valuei in tokens.items():
            #     sample_i[keyi] = valuei[::10+i, :]
            # ori_samples.append(sample_i)

            last_prompt_sample_i = (tokens_list[i]["labels"] == -100).sum(dim=-1) - 1
            ft_loss_sample_i = self.__cal_ft_loss(tokens_list[i], last_prompt_sample_i)
            act_loss_sample_i = self.__cal_activation_loss(self.get_adapter_layer().original_layer_output, self.get_adapter_layer().new_weight_layer_output,
                    config=config, act_mask=act_masks[i], deact_mask=deact_masks[i])
            loss_sampele_i = ft_loss_sample_i + act_loss_sample_i
            loss_sample.append(ft_loss_sample_i + act_loss_sample_i)

            loss_sampele_i.backward()

            # param = getattr(layer, layer_parts[-1])
            grad = self.get_adapter_layer().new_weight.grad

            sample_gradient = torch.norm(grad).item()
            gradient_list.append((i, sample_gradient))
            all_grad.append(grad)

        gradient_list.sort(key=lambda x: x[1], reverse=True)

        sorted_samples = [tokens_list[i] for i, _ in gradient_list]
        # sorted_gradients = [grad for _, grad in gradient_list]

        merged_samples = []
        processed_samples = set()

        for i in range(len(sorted_samples)):
            if i in processed_samples:
                continue

            # grad = gradient_list[i]
            merged_sample = [sorted_samples[i]]  # 保存合并的样本
            processed_samples.add(i)

            # 找出与当前样本相似度高的样本
            for j in range(i + 1, max(i+10, len(sorted_samples))):
                if j in processed_samples:
                    continue

                # _, other_grad = gradient_list[j]
                sim = cosine_similarity(all_grad[i], all_grad[j])

                if sim > 0.5:
                    merged_sample.append(sorted_samples[j])
                    processed_samples.add(j)

            # 合并相似样本后的编辑操作
            # 你可以根据合并后的样本来进行进一步的编辑操作
            merged_samples.append(merge_samples(merged_sample))  # `merge_samples`是你合并样本后的具体操作

        return sorted_samples

    def sort_samples_grad(self, config, tokens_list, act_masks=None, deact_masks=None):
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        loss_sample = []
        gradient_list = []
        # ori_samples = []

        for i in range(len(tokens_list)):
            self.model.zero_grad()

            # sample_i = {}
            # for keyi, valuei in tokens.items():
            #     sample_i[keyi] = valuei[::10+i, :]
            # ori_samples.append(sample_i)

            last_prompt_sample_i = (tokens_list[i]["labels"] == -100).sum(dim=-1) - 1
            ft_loss_sample_i = self.__cal_ft_loss(tokens_list[i], last_prompt_sample_i)
            act_loss_sample_i = self.__cal_activation_loss(self.get_adapter_layer().original_layer_output, self.get_adapter_layer().new_weight_layer_output,
                    config=config, act_mask=act_masks[i], deact_mask=deact_masks[i])
            loss_sampele_i = ft_loss_sample_i + act_loss_sample_i
            loss_sample.append(ft_loss_sample_i + act_loss_sample_i)

            loss_sampele_i.backward()

            # param = getattr(layer, layer_parts[-1])
            grad = self.get_adapter_layer().new_weight.grad

            sample_gradient = torch.norm(grad).item()
            gradient_list.append((i, sample_gradient))

        gradient_list.sort(key=lambda x: x[1], reverse=True) #
        sorted_samples = [tokens_list[i] for i, _ in gradient_list]
        return sorted_samples

    def build_sim_matrix_loop(self, gradient_vectors, fisher_masks):
        """内存优化的循环计算版本（支持列表输入）"""
        # 将输入统一转换为列表形式，避免显存占用
        if isinstance(gradient_vectors, torch.Tensor):
            grad_list = [gradient_vectors[i] for i in range(gradient_vectors.size(0))]
        else:
            grad_list = gradient_vectors

        n = len(grad_list)
        device = grad_list[0].device

        # 同样处理fisher_masks
        if isinstance(fisher_masks, torch.Tensor):
            fisher_list = [fisher_masks[i] for i in range(n)]
        else:
            fisher_list = fisher_masks

        assert len(fisher_list) == n, "gradient_vectors和fisher_masks长度不一致"

        sim_matrix = torch.zeros((n, n), device=device)
        eps = 1e-8

        with torch.no_grad():
            for i in range(n):
                grad_i = grad_list[i]
                mask_i = fisher_list[i]

                for j in range(i, n):
                    grad_j = grad_list[j]
                    mask_j = fisher_list[j]

                    # 计算并集mask
                    union_mask = torch.logical_or(mask_i, mask_j)

                    # 向量化计算
                    masked_i = grad_i * union_mask
                    masked_j = grad_j * union_mask

                    # 计算相似度
                    dot_product = torch.dot(masked_i, masked_j)
                    norm_product = torch.norm(masked_i) * torch.norm(masked_j)

                    # 对称赋值
                    if norm_product > eps:
                        sim_val = dot_product / norm_product
                        sim_matrix[i, j] = sim_val
                        sim_matrix[j, i] = sim_val
                    else:
                        sim_matrix[i, j] = 0.0
                        sim_matrix[j, i] = 0.0

                    # 及时释放中间变量
                    del union_mask, masked_i, masked_j, dot_product, norm_product

                # 定期清空缓存
                if i % 100 == 0:
                    torch.cuda.empty_cache()

        return sim_matrix

    def balanced_clustering(self, distance_matrix,
                        n_clusters_ori=10,
                        quantile=95):
        """
        自适应层次聚类 + 大簇二次分裂
        参数：
            distance_matrix: 预计算的距离矩阵
            size_threshold : 触发二次分裂的簇大小阈值
            quantile       : 用于动态阈值的分位数（0-100）
        """

        size_threshold = len(distance_matrix)//n_clusters_ori

        # 阶段1：动态阈值层次聚类
        dynamic_threshold = np.percentile(distance_matrix, quantile)
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity='precomputed',
            linkage='complete',
            distance_threshold=dynamic_threshold
        )
        labels = model.fit_predict(distance_matrix)

        del model, dynamic_threshold
        # labels = DBSCAN(
        #     metric='precomputed',
        #     eps=0.5
        # ).fit_predict(distance_matrix)

        # 阶段2：大簇二次分裂
        # unique_labels, counts = np.unique(labels, return_counts=True)
        # current_max_label = np.max(labels)

        # for cluster_id in unique_labels:
        #     if counts[cluster_id] > size_threshold:
        #         # 提取大簇样本的距离子矩阵
        #         mask = (labels == cluster_id)
        #         sub_dist = distance_matrix[mask][:, mask]

        #         # 自动确定分裂数量（至少分裂为2个）
        #         split_num = max(2, counts[cluster_id] // size_threshold)

        #         # 子聚类
        #         sub_labels = KMeans(n_clusters=split_num).fit_predict(sub_dist)

        #         # 更新标签（偏移原始标签范围）
        #         labels[mask] = sub_labels + current_max_label + 1
        #         current_max_label = np.max(labels)
        if self.config.subincluster:
            le = LabelEncoder()

            while True:
                unique_labels, counts = np.unique(labels, return_counts=True)
                big_clusters = unique_labels[counts > size_threshold]
                if not big_clusters.size:
                    break

                current_max_label = np.max(labels) if labels.size else 0
                for cluster_id in big_clusters:
                    mask = labels == cluster_id
                    sub_dist = distance_matrix[mask][:, mask]
                    cluster_size = counts[cluster_id]
                    split_num = max(2, cluster_size // size_threshold)

                    # 使用层次聚类分裂
                    sub_model = AgglomerativeClustering(
                        n_clusters=split_num,
                        affinity='precomputed',
                        linkage='complete'
                    )
                    sub_labels = sub_model.fit_predict(sub_dist)

                    # sub_labels = KMeans(n_clusters=split_num).fit_predict(sub_dist)

                    # 更新标签，避免冲突
                    new_labels = sub_labels + current_max_label + 1
                    labels[mask] = new_labels
                    current_max_label = np.max(labels)

                labels = le.fit_transform(labels)  # 例如原始标签[5,3,8] -> 转换为[0,1,2]
                del sub_dist, sub_labels, new_labels, sub_model


        linear_path = None
        if not self.config.dynamic_cluster:

            unique_labels_remapped = np.unique(labels)
            n_clusters = len(unique_labels_remapped)

            # 获取每个簇的样本索引
            cluster_indices = [np.where(labels == i)[0] for i in unique_labels_remapped]

            # 计算簇间平均距离矩阵
            cluster_dist = np.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    sub_dist = distance_matrix[cluster_indices[i]][:, cluster_indices[j]]
                    avg_dist = np.mean(sub_dist)
                    cluster_dist[i, j] = avg_dist
                    cluster_dist[j, i] = avg_dist  # 对称矩阵

            scaled_matrix = (cluster_dist * 1e6).astype(int).tolist()
            negative_matrix = [[-x for x in row] for row in scaled_matrix]
            path = elkai.solve_int_matrix(negative_matrix)

            # 将环路转换为线性路径（去除最大边）
            linear_path = path[:-1] if path[0] == path[-1] else path
            del cluster_indices, cluster_dist, scaled_matrix, path

            # # 将结果存储为类属性（或直接返回）
            # self.cluster_dist = cluster_dist
            # self.cluster_order = sorted_cluster_labels

        return labels, linear_path

    def cluster_samples(self, gradient_vectors, fisher_masks, n_clusters=10):
        """适配PyTorch向量化计算的聚类函数"""
        # 1. 计算相似度矩阵（保持设备一致）
        sim_matrix = self.build_sim_matrix_loop(gradient_vectors, fisher_masks)

        # 2. 设备感知处理
        device = sim_matrix.device
        if device.type == 'cuda':
            # 异步传输优化：提前pin内存加速CPU-GPU传输
            cpu_sim = sim_matrix.detach().cpu().numpy()
        else:
            cpu_sim = sim_matrix.numpy()

        # 3. 处理数值稳定性
        np.nan_to_num(cpu_sim, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)

        # 4. 构建距离矩阵（考虑负相似度）
        distance_matrix = 1 - cpu_sim#np.abs(cpu_sim)  # 使用绝对值处理负相关

        # 5. 内存优化：使用上三角压缩存储
        # condensed_dist = squareform(distance_matrix, checks=False)

        # 6. 分层聚类（支持大规模数据选项）
        # clustering = AgglomerativeClustering(
        #     n_clusters=n_clusters,
        #     affinity='precomputed',
        #     linkage='average',
        #     compute_full_tree='auto',
        # )

        # return clustering.fit_predict(distance_matrix)
        labels, linear_path = self.balanced_clustering(distance_matrix, n_clusters_ori=n_clusters, quantile=70)
        del  distance_matrix, sim_matrix
        return labels, linear_path

    def cluster_samples_grad(self, config, tokens_list, n_clusters=10, act_masks=None, deact_masks=None):
        # 启用训练模式和编辑模式
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()


        # gradient_vectors = []
        # indices = []
        # fisher_masks = []

        # for i in range(len(tokens_list)):
        #     self.model.zero_grad()

        #     # 计算损失
        #     last_prompt_idx = (tokens_list[i]["labels"] == -100).sum(dim=-1) - 1
        #     ft_loss = self.__cal_ft_loss(tokens_list[i], last_prompt_idx)
        #     act_loss = self.__cal_activation_loss(
        #         self.get_adapter_layer().original_layer_output,
        #         self.get_adapter_layer().new_weight_layer_output,
        #         config=config,
        #         act_mask=act_masks[i] if act_masks else None,
        #         deact_mask=deact_masks[i] if deact_masks else None
        #     )
        #     total_loss = ft_loss + act_loss

        #     # 反向传播获取梯度
        #     total_loss.backward()

        #     # 获取梯度矩阵并展平
        #     grad = self.get_adapter_layer().new_weight.grad
        #     grad_vector = grad.detach().cpu().view(-1)  # 展平为向量

        #     grad_threshold = grad_vector.kthvalue(int(0.5 * grad.numel())).values
        #     grad_mask = (grad_vector >= grad_threshold).float()

        #     fisher = torch.mul(grad_vector, grad_mask).pow(2)
        #     fisher_threshold = fisher.kthvalue(int(0.5 * fisher.numel())).values
        #     fisher_mask = (fisher >= fisher_threshold).float()

        #     gradient_vectors.append(grad_vector)
        #     indices.append(i)
        #     fisher_masks.append(fisher_mask)

        # 转换为numpy数组并进行标准化
        # X = np.stack(gradient_vectors)
        # X = StandardScaler().fit_transform(X)

        # pca = PCA(n_components=0.99, svd_solver='auto')
        # X = pca.fit_transform(X)

        # K-means聚类
        # kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)
        # kmeans = DBSCAN(eps=0.05, min_samples=5).fit(X)

        adapter_layer = self.get_adapter_layer()
        param_shape = adapter_layer.new_weight.shape  # 直接获取参数形状

        # 计算展平后的维度
        grad_dim = torch.prod(torch.tensor(param_shape)).item()

        # 设备一致性处理
        device = adapter_layer.new_weight.device  # 直接从参数获取设备信息
        # dtype = adapter_layer.new_weight.dtype    # 获取参数数据类型

        scale = 0.01

        # 预分配存储张量（优化内存布局）
        n_samples = len(tokens_list)
        n_clusters=n_samples//5
        gradient_vectors = torch.zeros(n_samples, grad_dim,
                                    device=device,
                                    dtype=torch.float16)
        fisher_masks = torch.zeros(n_samples, grad_dim,
                                device=device,
                                dtype=torch.bool)  # 使用bool类型节省空间
        indices = []

        for i in range(n_samples):
            self.model.zero_grad()

            # 计算损失（保持原有逻辑）
            last_prompt_idx = (tokens_list[i]["labels"] == -100).sum(dim=-1) - 1
            ft_loss = self.__cal_ft_loss(tokens_list[i], last_prompt_idx)
            act_loss = self.__cal_activation_loss(
                adapter_layer.original_layer_output,
                adapter_layer.new_weight_layer_output,
                config=config,
                act_mask=act_masks[i] if act_masks else None,
                deact_mask=deact_masks[i] if deact_masks else None
            )
            total_loss = ft_loss + act_loss

            # 反向传播获取梯度
            total_loss.backward(retain_graph=False)
            del total_loss, ft_loss, act_loss

            # 安全获取梯度（确保存在梯度）
            if adapter_layer.new_weight.grad is None:
                raise RuntimeError("梯度未正确计算，请检查反向传播过程")

            # 直接操作梯度张量（避免内存复制）
            grad = adapter_layer.new_weight.grad.detach()
            grad_vector = grad.view(-1)  # 保持视图而非复制

            # 计算mask（保持原位操作）
            k = int(0.5 * grad.numel())
            grad_threshold = grad_vector.kthvalue(k).values
            grad_mask = grad_vector >= grad_threshold

            # Fisher计算优化
            fisher = (grad_vector * grad_mask).square()
            fisher_threshold = fisher.kthvalue(k).values

            # quantized_grad_vector = torch.clamp(
            #     torch.round(grad_vector / scale),
            #     min=-128, max=127
            # ).to(torch.int8)

            # 直接填充预分配张量
            gradient_vectors[i] = grad_vector.half()
            fisher_masks[i] = fisher >= fisher_threshold

            indices.append(i)
            adapter_layer.new_weight.grad = None

            del grad_vector, fisher

        cluster_labels, ordered_clusters = self.cluster_samples(gradient_vectors, fisher_masks, n_clusters=n_clusters)
        le = LabelEncoder()
        labels_remapped = le.fit_transform(cluster_labels)  # 例如原始标签[5,3,8] -> 转换为[0,1,2]
        num_clusters = len(le.classes_)

        del gradient_vectors,fisher_masks
        # 按聚类结果分组样本
        clustered_samples = [[] for _ in range(num_clusters)]
        clu_act_masks = [[] for _ in range(num_clusters)]
        clu_deact_masks = [[] for _ in range(num_clusters)]
        clu_index = [[] for _ in range(num_clusters)]
        for idx, label in zip(indices, labels_remapped):
            clustered_samples[label].append(tokens_list[idx])
            clu_act_masks[label].append(act_masks[idx])
            clu_deact_masks[label].append(deact_masks[idx])
            clu_index[label].append(idx)

        if config.maxsimseq:
            clustered_samples = [clustered_samples[i] for i in ordered_clusters]
            clu_act_masks = [clu_act_masks[i] for i in ordered_clusters]
            clu_deact_masks = [clu_deact_masks[i] for i in ordered_clusters]
            clu_index = [clu_index[i] for i in ordered_clusters]

        return clustered_samples, clu_act_masks, clu_deact_masks, clu_index

    def cluster_dynamic_grad(self, config, tokens_list, n_clusters=10, batch_size=100,
                            act_masks=None, deact_masks=None, similarity_threshold=0.7):
        # 启用训练模式和编辑模式
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()

        adapter_layer = self.get_adapter_layer()
        param_shape = adapter_layer.new_weight.shape
        grad_dim = torch.prod(torch.tensor(param_shape)).item()
        device = adapter_layer.new_weight.device

        # 初始化簇存储结构
        cluster_centers = []          # 存储每个簇的梯度均值（设备内存）
        cluster_fisher_masks = []     # 存储每个簇的Fisher掩码均值
        clustered_samples = []        # 样本存储
        clu_act_masks = [] if act_masks else None
        clu_deact_masks = [] if deact_masks else None
        clu_index = []

        batch_samples_max = len(tokens_list)//n_clusters


        # 分批次处理
        for batch_idx in range(0, len(tokens_list), batch_size):
            batch_tokens = tokens_list[batch_idx:batch_idx+batch_size]
            current_batch_size = len(batch_tokens)

            gradient_vectors = torch.zeros(current_batch_size, grad_dim,
                                        device=device,
                                        dtype=torch.float16)
            fisher_masks = torch.zeros(current_batch_size, grad_dim,
                                    device=device,
                                    dtype=torch.bool)

            # 步骤1: 计算当前批次样本的特征
            # batch_grads, batch_fishers = [], []
            for idx_in_batch in range(current_batch_size):
                self.model.zero_grad()

                # 计算梯度特征
                token_data = batch_tokens[idx_in_batch]
                last_prompt_idx = (token_data["labels"] == -100).sum(dim=-1) - 1
                ft_loss = self.__cal_ft_loss(token_data, last_prompt_idx)
                act_loss = self.__cal_activation_loss(
                    adapter_layer.original_layer_output,
                    adapter_layer.new_weight_layer_output,
                    config=config,
                    act_mask=act_masks[batch_idx+idx_in_batch] if act_masks else None,
                    deact_mask=deact_masks[batch_idx+idx_in_batch] if deact_masks else None
                )
                (ft_loss + act_loss).backward()

                # 提取特征
                grad = adapter_layer.new_weight.grad.detach().view(-1)
                k = int(0.5 * grad.numel())
                grad_threshold = grad.kthvalue(k).values
                grad_mask = grad >= grad_threshold
                fisher = (grad * grad_mask).square()
                fisher_threshold = fisher.kthvalue(k).values
                fisher_mask = (fisher >= fisher_threshold)

                gradient_vectors[idx_in_batch] = grad.half()
                fisher_masks[idx_in_batch] = fisher_mask
                # batch_grads.append(grad)
                # batch_fishers.append(fisher_mask)
                adapter_layer.new_weight.grad = None
                del grad, fisher

            # 步骤2: 动态分配样本到已有簇或创建新簇
            if not cluster_centers:  # 首个批次
                # 初始聚类
                # grad_tensor = torch.stack(batch_grads)
                # fisher_tensor = torch.stack(batch_fishers)
                labels, _ = self.cluster_samples(gradient_vectors, fisher_masks, n_clusters)

                unique_labels_remapped = np.unique(labels)
                # n_clusters = len(unique_labels_remapped)

                # 初始化簇中心
                for cluster_id in range(len(unique_labels_remapped)):
                    mask = (torch.tensor(labels) == cluster_id)
                    center_grad = gradient_vectors[mask].mean(dim=0)
                    center_fisher = fisher_masks[mask].float().mean(dim=0) > 0.5
                    cluster_centers.append(center_grad)
                    cluster_fisher_masks.append(center_fisher)

                    # 存储样本
                    clustered_samples.append([batch_tokens[i] for i in torch.where(mask)[0].tolist()])
                    if act_masks:
                        clu_act_masks.append([act_masks[batch_idx+i] for i in torch.where(mask)[0].tolist()])
                    if deact_masks:
                        clu_deact_masks.append([deact_masks[batch_idx+i] for i in torch.where(mask)[0].tolist()])
                    clu_index.append([batch_idx+i for i in torch.where(mask)[0].tolist()])

                del gradient_vectors, fisher_masks
            else:  # 后续批次
                if len(cluster_centers)>=2:
                    center_tensor = torch.stack(cluster_centers)
                    center_fisher_mask = torch.stack(cluster_fisher_masks)
                    sim_matrix = self.build_sim_matrix_loop(center_tensor, center_fisher_mask)
                    np_sim = sim_matrix.cpu().numpy()
                    np.fill_diagonal(np_sim, -np.inf)
                    similarity_threshold = np.nanmax(np_sim)
                    # min_cluster_distance = 1 - max_sim
                    # similarity_threshold = max(0.1, max_sim)

                    del center_tensor, sim_matrix, np_sim#, max_sim  , min_cluster_distance
                else:
                    similarity_threshold = 0.5

                for sample_idx in range(current_batch_size):
                    grad = gradient_vectors[sample_idx]
                    fisher = fisher_masks[sample_idx]

                    # 计算与所有簇中心的相似度
                    similarities = []
                    for center, mask in zip(cluster_centers, cluster_fisher_masks):
                        # 应用Fisher掩码计算相似度
                        masked_grad = grad * mask.float()
                        masked_center = center * mask.float()

                        dot_product = torch.dot(masked_grad, masked_center)
                        norm_product = torch.norm(masked_grad) * torch.norm(masked_center)
                        sim = dot_product / (norm_product + 1e-8)  # 等效余弦相似度

                        # sim = F.cosine_similarity(masked_grad, masked_center, dim=0)
                        similarities.append(sim.item())
                        del masked_grad, masked_center, dot_product, norm_product, sim

                    # 寻找最佳匹配簇
                    max_sim = max(similarities) if similarities else 0
                    best_cluster = similarities.index(max_sim) if similarities else -1

                    if max_sim >= similarity_threshold and len(clustered_samples[best_cluster])<=batch_samples_max:  # 加入已有簇
                        clustered_samples[best_cluster].append(batch_tokens[sample_idx])
                        if act_masks:
                            clu_act_masks[best_cluster].append(act_masks[batch_idx+sample_idx])
                        if deact_masks:
                            clu_deact_masks[best_cluster].append(deact_masks[batch_idx+sample_idx])
                        clu_index[best_cluster].append(batch_idx + sample_idx)

                        # 更新簇中心（增量平均）
                        n = len(clustered_samples[best_cluster])
                        cluster_centers[best_cluster] = (cluster_centers[best_cluster] * (n-1) + grad) / n
                        cluster_fisher_masks[best_cluster] = (cluster_fisher_masks[best_cluster] * (n-1) + fisher) > (n//2)
                    else:  # 创建新簇
                        cluster_centers.append(grad)
                        cluster_fisher_masks.append(fisher)
                        clustered_samples.append([batch_tokens[sample_idx]])
                        if act_masks:
                            clu_act_masks.append([act_masks[batch_idx+sample_idx]])
                        if deact_masks:
                            clu_deact_masks.append([deact_masks[batch_idx+sample_idx]])
                        clu_index.append([batch_idx + sample_idx])

                del gradient_vectors, fisher_masks

            # 步骤3: 合并相似簇（使用动态阈值）
            # if len(cluster_centers) > 1:
            #     # 计算动态阈值
            #     # center_tensor = torch.stack(cluster_centers)
            #     # sim_matrix = self.build_sim_matrix_loop(center_tensor, None)
            #     # np_sim = sim_matrix.cpu().numpy()
            #     # np.fill_diagonal(np_sim, -np.inf)
            #     # max_sim = np.nanmax(np_sim)
            #     # min_cluster_distance = 1 - max_sim
            #     # dynamic_threshold = 1 - 0.5 * min_cluster_distance

            #     # 合并逻辑
            #     merge_mask = (sim_matrix > similarity_threshold).triu(diagonal=1)
            #     merge_pairs = torch.where(merge_mask)

            #     # 按相似度从高到低合并
            #     sorted_pairs = sorted(zip(merge_pairs[0].tolist(), merge_pairs[1].tolist()),
            #                         key=lambda x: -sim_matrix[x[0], x[1]])

            #     merged = set()
            #     for i,j in sorted_pairs:
            #         if i not in merged and j not in merged and i < j:
            #             # 合并簇i和j
            #             n_i = len(clustered_samples[i])
            #             n_j = len(clustered_samples[j])
            #             new_center = (cluster_centers[i]*n_i + cluster_centers[j]*n_j)/(n_i+n_j)
            #             cluster_centers[i] = new_center
            #             del cluster_centers[j]

            #             clustered_samples[i].extend(clustered_samples[j])
            #             del clustered_samples[j]

            #             # 更新其他数据...（同原逻辑）
            #             merged.update({i,j})

        # 最终排序处理（保持原有逻辑）
        if config.maxsimseq:
            # 计算簇中心模长排序
            final_sim_matrix = self.build_sim_matrix_loop(cluster_centers, cluster_fisher_masks)

            scaled_matrix = (final_sim_matrix * 1e6).int().tolist()
            negative_matrix = [[-x for x in row] for row in scaled_matrix]
            path = elkai.solve_int_matrix(negative_matrix)

            ordered_clusters = path[:-1] if path[0] == path[-1] else path

            clustered_samples = [clustered_samples[i] for i in ordered_clusters]
            if act_masks:
                clu_act_masks = [clu_act_masks[i] for i in ordered_clusters]
            if deact_masks:
                clu_deact_masks = [clu_deact_masks[i] for i in ordered_clusters]
            clu_index = [clu_index[i] for i in ordered_clusters]
        elif config.cluster_sort == 'greed':
            # cluster_norms = torch.norm(cluster_centers, dim=1)  # 计算每个簇的模长
            final_sim_matrix = self.build_sim_matrix_loop(cluster_centers, cluster_fisher_masks)
            cluster_norms = []
            for c in cluster_centers:
                # 处理不同形状的张量
                flattened = c.flatten().clone().detach()  # 确保无梯度影响
                norm = torch.norm(flattened).item()       # 转为标量值
                cluster_norms.append(norm)

            remaining_indices = list(range(len(cluster_centers)))

            # 1. 选择模长最小的作为第一个簇
            current_idx = cluster_norms.index(min(cluster_norms))  # 直接找最小值的索引
            ordered_clusters = [current_idx]
            remaining_indices.remove(current_idx)

            # 2. 贪心选择后续簇
            while remaining_indices:
                # 获取当前簇与剩余簇的相似度
                similarities = final_sim_matrix[current_idx, remaining_indices]

                # 找到相似度最小的（假设相似度矩阵值越大表示越相似）
                next_rel_idx = int(torch.argmax(similarities))
                current_idx = remaining_indices[next_rel_idx]

                ordered_clusters.append(current_idx)
                remaining_indices.pop(next_rel_idx)

            clustered_samples = [clustered_samples[i] for i in ordered_clusters]
            if act_masks:
                clu_act_masks = [clu_act_masks[i] for i in ordered_clusters]
            if deact_masks:
                clu_deact_masks = [clu_deact_masks[i] for i in ordered_clusters]
            clu_index = [clu_index[i] for i in ordered_clusters]
            del cluster_centers, final_sim_matrix, cluster_fisher_masks
        elif config.randomseq:
            ordered_clusters = list(range(len(cluster_centers)))
            random.shuffle(ordered_clusters)

            clustered_samples = [clustered_samples[i] for i in ordered_clusters]
            if act_masks:
                clu_act_masks = [clu_act_masks[i] for i in ordered_clusters]
            if deact_masks:
                clu_deact_masks = [clu_deact_masks[i] for i in ordered_clusters]
            clu_index = [clu_index[i] for i in ordered_clusters]


        return clustered_samples, clu_act_masks, clu_deact_masks, clu_index

    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        # for retrieve ##
        global edit_history
        global merge_group_edit_history

        edit_history.append([{f"{k1}" : v1.to('cpu') for k1, v1 in tokens.items()}, False])
        # for retrieve ##
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_lora_parameter_tunable()

        sample_id = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt")
        if not config.sensemask and sample_id % self.config.save_freq == 0:
            self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)
        # if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % 10 == 0:
        #     # self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)
        #     self.get_adapter_layer().generate_non_overlapping_mask(0.2)

        # --- train Wise value ---
        loss_meter = EarlyStopMeter()

        current_start = 0
        # config.n_iter = 300
        self.model.zero_grad()
        # if sample_id==1:
        #     self.get_adapter_layer().moe_weight[0].grad = None
        for i in range(config.n_iter):

            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([self.get_adapter_layer().lora_weight.B], config.edit_lr, weight_decay=1e-5)

            ft_loss = self.__cal_ft_loss(tokens, last_prompt_token_loc)

            if i == 0:
                train_weight_output = self.get_adapter_layer().new_weight_layer_output
            else:
                train_weight_output = self.get_adapter_layer().moe_weight_layer_output

            # act_loss = self.__cal_activation_loss(self.get_adapter_layer().original_layer_output, train_weight_output,
            #                                     config=config, act_mask=act_mask, deact_mask=deact_mask)
            # act_loss = config.act_loss_weight*act_loss.to(ft_loss.device)
            regu_loss = 0
            if not self.get_adapter_layer().create_new_weight_flag and i != 0 and self.get_adapter_layer().ori_rel_outs is not None and not self.config.drop_regu:
                regu_loss = self.__cal_regu_loss(self.get_adapter_layer().cluster_rel_activation[self.get_adapter_layer().select_weight_id],
                                                self.get_adapter_layer().ori_rel_outs)

            # loss = ft_loss + regu_loss*5#+ act_loss

            loss = ft_loss + regu_loss * self.config.regu_loss_weight#+ act_loss


            # if self.config.use_chat_template:
                # loss = ft_loss + regu_loss * 2

            if loss is None:
                raise RuntimeError("Loss was not computed in any branch.")

            optimizer.zero_grad()

            loss.backward()

            def trans_dtype_by_str(x, str):
                if str == '':
                    return x
                if str == 'int8':
                    return (x * (120 / x.abs().max().item())).to(getattr(torch, str))
                return x.to(getattr(torch, str))

            if loss_meter.stop():
                if config.sensemask:
                    self.get_adapter_layer().save_moe_activation()
                else:
                    self.get_adapter_layer().save_editing_activation()  # add last gradient
                self.get_adapter_layer().moe_grad = False
                print(f'The number of MOE layers: {len(self.get_adapter_layer().moe_weight)}')
                if act_mask[0] is not None:
                    # 1. 参数量：lora rank
                    #
                    # moe_activations = torch.mean(self.get_adapter_layer().knowledge_act*act_mask[0][0].unsqueeze(-1), dim=0)
                    moe_activations = torch.mean(self.get_adapter_layer().knowledge_act[0][act_mask[0][0].to(bool)], dim=0)
                    moe_activations = trans_dtype_by_str(moe_activations, self.config.activations_dtype)
                    rel_activations = torch.mean(self.get_adapter_layer().knowledge_act[0], dim=0)
                    # rel_activations = self.get_adapter_layer().knowledge_act[0][-1]
                    self.get_adapter_layer().cluster_rel_activation[self.get_adapter_layer().select_weight_id].append(rel_activations)

                    subself_dist = torch.cosine_similarity(moe_activations, rel_activations, dim=0)
                    loc_dist = torch.cosine_similarity(moe_activations, torch.mean(self.get_adapter_layer().knowledge_act[-1],dim=0), dim=0)
                    # loc_dist = torch.cosine_similarity(moe_activations, self.get_adapter_layer().knowledge_act[-1][-1], dim=0)
                    sub_bound = loc_dist + self.config.bound_ratio*(subself_dist - loc_dist)

                    rel_dist = torch.cosine_similarity(rel_activations, torch.mean(self.get_adapter_layer().knowledge_act[-1],dim=0), dim=0)
                    # rel_dist = torch.cosine_similarity(rel_activations, self.get_adapter_layer().knowledge_act[-1][-1], dim=0)
                    rel_bound = rel_dist + self.config.bound_ratio*(1- rel_dist)

                    self.get_adapter_layer().cluster_rel_bound[self.get_adapter_layer().select_weight_id].append(rel_bound)
                else:
                    if self.config.act_key_last:
                        moe_activations = self.get_adapter_layer().knowledge_act[0][-1]
                    else:
                        moe_activations = torch.mean(self.get_adapter_layer().knowledge_act[0], dim=0)

                    moe_activations = trans_dtype_by_str(moe_activations, self.config.activations_dtype)

                    sub_dist = torch.cosine_similarity(moe_activations, torch.mean(self.get_adapter_layer().knowledge_act[-1],dim=0), dim=0)
                    sub_bound = sub_dist + self.config.bound_ratio*(1- sub_dist)
                # self.get_adapter_layer().update_cluster_activation(moe_activations)

                self.get_adapter_layer().cluster_activation[self.get_adapter_layer().select_weight_id].append(moe_activations)
                if self.config.use_chat_template:
                    try:
                        self.get_adapter_layer().cluster_bound[self.get_adapter_layer().select_weight_id].append(rel_bound)
                    except:
                        self.get_adapter_layer().cluster_bound[self.get_adapter_layer().select_weight_id].append(sub_bound)
                else:
                    self.get_adapter_layer().cluster_bound[self.get_adapter_layer().select_weight_id].append(sub_bound)

                for cluster_i in self.get_adapter_layer().moe_cluster:
                    print(cluster_i['sample_id'], flush=True)

                last_grad = self.get_adapter_layer().moe_weight[self.get_adapter_layer().select_weight_id].B.grad.detach().view(-1)
                self.get_adapter_layer().moe_cluster[self.get_adapter_layer().select_weight_id].update({'history_grad':last_grad})
                break
            if i == config.n_iter - 1:
                if config.sensemask:
                    self.get_adapter_layer().save_moe_activation()
                else:
                    self.get_adapter_layer().save_editing_activation()  # add last gradient
                self.get_adapter_layer().moe_grad = False
                print(f'The number of MOE layers: {len(self.get_adapter_layer().moe_weight)}')
                if act_mask[0] is not None:
                    moe_activations = torch.mean(self.get_adapter_layer().knowledge_act[0][act_mask[0][0].to(bool)], dim=0)
                    moe_activations = trans_dtype_by_str(moe_activations, self.config.activations_dtype)
                    rel_activations = torch.mean(self.get_adapter_layer().knowledge_act[0], dim=0)
                    # rel_activations = self.get_adapter_layer().knowledge_act[0][-1]
                    self.get_adapter_layer().cluster_rel_activation[self.get_adapter_layer().select_weight_id].append(rel_activations)

                    subself_dist = torch.cosine_similarity(moe_activations, rel_activations, dim=0)
                    loc_dist = torch.cosine_similarity(moe_activations, torch.mean(self.get_adapter_layer().knowledge_act[-1],dim=0), dim=0)
                    # loc_dist = torch.cosine_similarity(moe_activations, self.get_adapter_layer().knowledge_act[-1][-1], dim=0)
                    sub_bound = loc_dist + self.config.bound_ratio*(subself_dist - loc_dist)

                    rel_dist = torch.cosine_similarity(rel_activations, torch.mean(self.get_adapter_layer().knowledge_act[-1],dim=0), dim=0)
                    # rel_dist = torch.cosine_similarity(rel_activations, self.get_adapter_layer().knowledge_act[-1][-1], dim=0)
                    rel_bound = rel_dist + self.config.bound_ratio*(1- rel_dist)

                    self.get_adapter_layer().cluster_rel_bound[self.get_adapter_layer().select_weight_id].append(rel_bound)
                else:
                    if self.config.act_key_last:
                        moe_activations = self.get_adapter_layer().knowledge_act[0][-1]
                    else:
                        moe_activations = torch.mean(self.get_adapter_layer().knowledge_act[0], dim=0)

                    moe_activations = trans_dtype_by_str(moe_activations, self.config.activations_dtype)

                    sub_dist = torch.cosine_similarity(moe_activations, torch.mean(self.get_adapter_layer().knowledge_act[-1],dim=0), dim=0)
                    sub_bound = sub_dist + self.config.bound_ratio*(1- sub_dist)
                # self.get_adapter_layer().update_cluster_activation(moe_activations)
                self.get_adapter_layer().cluster_activation[self.get_adapter_layer().select_weight_id].append(moe_activations)

                if self.config.use_chat_template:
                    try:
                        self.get_adapter_layer().cluster_bound[self.get_adapter_layer().select_weight_id].append(rel_bound)
                    except:
                        self.get_adapter_layer().cluster_bound[self.get_adapter_layer().select_weight_id].append(sub_bound)
                else:
                    self.get_adapter_layer().cluster_bound[self.get_adapter_layer().select_weight_id].append(sub_bound)

                last_grad = self.get_adapter_layer().moe_weight[self.get_adapter_layer().select_weight_id].B.grad.detach().view(-1)
                self.get_adapter_layer().moe_cluster[self.get_adapter_layer().select_weight_id].update({'history_grad':last_grad})

                for cluster_i in self.get_adapter_layer().moe_cluster:
                    print(cluster_i['sample_id'], flush=True)
            # if config.sensemask and i == 0:
            #     loss.backward(retain_graph=True)
            # else:
            #     loss.backward()

            if config.sensemask and i == 0:
                grad = self.get_adapter_layer().lora_weight.B.grad.detach().view(-1)
                grad_A = self.get_adapter_layer().lora_weight.A.grad.detach().view(-1)
                # k = int(0.8 * grad.numel())
                # # if sample_id == 6:
                # #     k = int(0.6 * grad.numel())
                # grad_threshold = grad.kthvalue(k).values
                # grad_mask = grad >= grad_threshold
                # fisher = (grad * grad_mask).square()
                # # k = int(0.75 * grad.numel())
                # fisher_threshold = fisher.kthvalue(k).values
                # fisher_mask = (fisher >= fisher_threshold)
                # fisher = grad.square()
                # k = int(0.75 * fisher.numel())
                # # if sample_id == 6:
                # #     k = int(0.6 * fisher.numel())
                # fisher_threshold = fisher.kthvalue(k).values
                fisher = grad.square()
                d = grad.numel()  # 参数总维度

                # 归一化Fisher熵计算
                prob = fisher / (fisher.sum() + 1e-8)
                max_entropy = torch.log(torch.tensor(d, dtype=torch.float))  # 最大可能熵值
                normalized_entropy = (-torch.sum(prob * torch.log(prob + 1e-8))) / max_entropy

                # 动态比例稳定计算
                grad_norm = grad.norm(p=2)
                if config.use_gamma_para:
                    dynamic_ratio = config.ga_para_0 + config.ga_para_1 * (1 - normalized_entropy)
                else:
                    dynamic_ratio = config.init_train_ratio * (0.6 + 2 * (1 - normalized_entropy))
                # 0.57 + 1.9 * (1 - H)



                # 安全阈值生成
                if config.use_gamma_para:
                    k = int(dynamic_ratio.clamp(0.0, 1.0) * d)
                else:
                    k = int(dynamic_ratio.clamp(0.7, 0.96) * d)  # 限制在50%-85%区间
                # k = int(config.init_train_ratio * d)
                if k != 0:
                    fisher_threshold = fisher.kthvalue(k).values  # 修正kthvalue索引
                    fisher_mask = (fisher >= fisher_threshold)
                else:
                    fisher_mask = torch.ones_like(fisher)
                print("### Real mask ratio", fisher_mask.sum() / fisher_mask.numel(), flush=True)
                # if sample_id % self.config.save_freq == 0:
                #     self.get_adapter_layer().generate_mask_sensegrad(fisher_mask)
                # self.get_adapter_layer().generate_non_overlapping_mask(0)

                self.get_adapter_layer().select_weight(sample_id, grad, fisher_mask, grad_A)
                self.model.zero_grad()
                select_weight_id = self.get_adapter_layer().select_weight_id
                self.get_adapter_layer().lora_weight.grad = None
                # optimizer.state = defaultdict(dict)
                if self.get_adapter_layer().create_new_weight_flag:
                    edit_lr = config.edit_lr
                else:
                    edit_lr = config.edit_lr*0.1
                if self.config.use_chat_template:
                    edit_lr /= 5
                optimizer = torch.optim.SGD([self.get_adapter_layer().moe_weight[select_weight_id].B, self.get_adapter_layer().moe_weight[select_weight_id].A], edit_lr, weight_decay=1e-5)
                # del loss
                # self.get_adapter_layer().update_cluster_center(grad, grad_mask)


                continue
            if config.grad_sensetive_area:
                self.get_adapter_layer().mask_moe_weight_gradient()

            torch.nn.utils.clip_grad_norm_([self.get_adapter_layer().moe_weight[select_weight_id].B], max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([self.get_adapter_layer().moe_weight[select_weight_id].A], max_norm=1.0)

            if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}", flush=True
                )
            else:
                if isinstance(ft_loss, list) and isinstance(regu_loss, list):
                    for ft_loss_i, regu_loss_i in zip(ft_loss, regu_loss):
                        print(
                            f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss_i.item(), 3)} + {np.round(regu_loss_i.item(), 3)}",
                            flush=True
                        )
                elif isinstance(ft_loss, Tensor) and isinstance(regu_loss, Tensor):
                    print(
                        f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(regu_loss.item(), 3)}",
                        flush=True
                    )
                else:
                    print(
                        f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(regu_loss, 3)}",
                        flush=True
                    )

            optimizer.step()
            loss_meter.update(loss.item())

            # if type(self.config.norm_constraint) is float:
            #     self.__norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
        #
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            self.get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            self.get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

        if self.config.merge_moe and editing_total_cnt % self.config.merge_moe_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            self.get_adapter_layer().merge_moe_weight_pair()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')
            print('length of memory is ', len(self.get_adapter_layer().moe_weight), '!!!!!!')

    def __norm_constraint(self, norm_constraint):
        lora_layer = self.get_adapter_layer().lora_weight
        original_B = lora_layer.B  # 假设LoRA层的B矩阵需要约束
        with torch.no_grad():
            lora_layer.B[...] = torch.clamp(
                lora_layer.B,
                min=original_B - norm_constraint,
                max=original_B + norm_constraint
            )

    def __cal_ft_loss(self, tokens, last_prompt_token_loc):
        if hasattr(self.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        bs = tokens["input_ids"].shape[0] - k
        logits = self.model(**tokens).logits
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = tokens['labels'][:-k, 1:].contiguous()


        # bs = tokens["input_ids"].shape[0]
        # logits = self.model(**tokens).logits
        # shift_logits = logits[:, :-1, :].contiguous()
        # shift_labels = tokens['labels'][:, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bs, -1)

        label_mask = torch.zeros_like(loss, dtype=torch.bool)

        for i, col_index in enumerate(last_prompt_token_loc[:-k]):
        # for i, col_index in enumerate(last_prompt_token_loc[:]):
            label_mask[i, col_index - 1:] = True

        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        del logits, shift_logits, shift_labels
        return ft_loss

    def __cal_regu_loss(self, rel_act_list, rel_ori_out):

        regu_loss = 0
        if len(rel_act_list) > 0:

            rel_act = torch.stack(rel_act_list)
            lora_i = self.get_adapter_layer().moe_weight[self.get_adapter_layer().select_weight_id]
            new_rel_out = lora_i(rel_act)

            mse_loss = nn.MSELoss()
            regu_loss = mse_loss(new_rel_out, rel_ori_out)
        # for (rel_act, ori_out) in zip(rel_act_list, rel_ori_out):
        #     lora_i = self.get_adapter_layer().moe_weight[self.get_adapter_layer().select_weight_id]
        #     new_rel_out = lora_i(rel_act)
        #     mse_loss = nn.MSELoss()
        #     regu_loss += mse_loss(new_rel_out, ori_out)

        return regu_loss

    def __cal_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        total_loss = []
        len_temp = original_layer_output.shape[0] / k - 1
        for i,act_mk in enumerate(act_mask):
            if act_mk is not None:
                in_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config,
                                    act_mask=act_mk)
                out_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config,
                                    act_mask=deact_mask[i])
            else:
                in_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config)
                if (i==k-1):
                    out_scope_dist = euc(original_layer_output[int(i-k):, ...], new_weight_layer_output[int(i-k):, ...], config)
                else:
                    out_scope_dist = euc(original_layer_output[int(i-k):int(i+1-k), ...], new_weight_layer_output[int(i-k):int(i+1-k), ...], config)

            loss = out_scope_dist.view(-1,1) - in_scope_dist + config.gamma
            loss2 = out_scope_dist - config.alpha
            loss3 = config.beta - in_scope_dist
            loss3 = torch.mean(loss3[loss3 > 0]) if min(loss3[loss3 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            loss2 = torch.mean(loss2[loss2 > 0]) if min(loss2[loss2 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            loss = torch.mean(loss[loss > 0]) if min(loss[loss > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            total_loss.append(loss + loss2 + loss3)
        return sum(total_loss) / len(total_loss)

    def __cal_memory_pos_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = 20 - in_scope_dist

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def __cal_memory_neg_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = in_scope_dist - 5

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def save(self, save_path):
        import os
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Save additional information, such as memory_weight, memory_mean_act, etc.
        additional_info = {
            'memory_weight': self.get_adapter_layer().memory_weight,
            'memory_mean_act': self.get_adapter_layer().memory_mean_act,
            'merge_cnt': self.get_adapter_layer().merge_cnt,
            'editing_mean_act': self.get_adapter_layer().editing_mean_act,
            'editing_total_cnt': self.get_adapter_layer().editing_total_cnt,
            'weight_mask': self.get_adapter_layer().weight_mask,
            # Add other variables that need to be saved
        }
        if hasattr(self.get_adapter_layer(), 'key_id') and self.get_adapter_layer().key_id is not None:
            additional_info['key_id'] = self.get_adapter_layer().key_id
        # Save all information to the file
        torch.save({
            'adapter_state_dict': self.get_adapter_layer().state_dict(),
            'config': self.config,
            'additional_info': additional_info,
            'edit_history': edit_history,
            'merge_group_edit_history': merge_group_edit_history
        }, save_path)

    def load(self, load_path):
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        # Load all previously saved information
        saved_data = torch.load(load_path)
        if hasattr(self.model.config, 'hidden_act'):
            saved_data['config'].hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            saved_data['config'].hidden_act = self.model.config.activation_function
        if saved_data['config'] != self.config:
            print("Warning: The loaded WISE config is different from the original config")

        # Restore the state dictionary of the WISE Adapter instance
        self.get_adapter_layer().load_state_dict(saved_data['adapter_state_dict'])
        # Restore additional information
        adapter_layer = self.get_adapter_layer()
        for key, value in saved_data['additional_info'].items():
            setattr(adapter_layer, key, value)

        # Restore editing history
        global edit_history, merge_group_edit_history
        edit_history = saved_data['edit_history']
        merge_group_edit_history = saved_data['merge_group_edit_history']
        print(f"Model configuration and WISE state loaded from {load_path}")

class WISEAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(WISEAdapter, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config
        self.new_weight = copy.deepcopy(self.weight)
        self.original_layer = copy.deepcopy(self.layer)

        in_dim = self.layer.weight.shape[1]  # 4096
        out_dim = self.layer.weight.shape[0]  # 11008

        self.lora_weight = LoRALayer(
            in_dim=in_dim,
            out_dim=out_dim,
            rank=config.lora_rank,
            alpha=config.adapter_scale  # 将缩放因子合并到LoRALayer
        ).to(self.device)

        self.memory_weight = []
        self.memory_mean_act = []
        if 'gpt2' in self.config.model_name:
            self.bias = self.layer.bias # For Conv1D
        else:
            self.bias = None
        self.merge_cnt = 0  # only for retrieve
        assert not self.weight.requires_grad, print('Original Layer can not be tunable....')

        self.used_mask = None

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False
        self.editing = False

        self.editing_mean_act = EditingMeanAct()
        self.editing_total_cnt = 0

        self.moe_weight = []
        self.moe_cluster = []
        self.select_weight_id = 0
        self.sim_threshold = config.sim_threshold
        self.moe_mean_act = []
        self.moe_grad = False
        self.cfl_threshold = config.cfl_threshold

        self.cluster_activation = []
        self.cluster_bound = []

        self.cluster_rel_activation = []
        self.cluster_rel_bound = []
        self.knowledge_act = None
        self.history_grad = None
        self.create_new_weight_flag = True
        self.ori_rel_outs = None

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def save_weight(self):
        self.memory_weight.append(copy.deepcopy(self.new_weight))
        self.new_weight = copy.deepcopy(self.original_layer.weight)
        if self.config.retrieve:
            self.memory_mean_act.append(copy.deepcopy(self.editing_mean_act))
            self.editing_mean_act = EditingMeanAct()

    def merge_weight(self):
        if self.config.save_freq is not None:  # for ties dare dare_ties
            if not self.config.retrieve:
                merge_alg = merge_dict[self.config.merge_alg]
                if self.original_layer.weight.equal(self.layer.weight):
                    cur_new_weight = merge_alg.execute([self.config.weights / len(self.memory_weight) for _ in range(len(self.memory_weight))], self.original_layer.weight, self.memory_weight, densities=self.config.densities)
                else:
                    cur_new_weight = merge_alg.execute([0.4 / len(self.memory_weight) for _ in range(len(self.memory_weight))] + [0.6], self.original_layer.weight, self.memory_weight + [self.layer.weight], densities=self.config.densities)
                self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                del self.memory_weight
                self.memory_weight = []
            else:
                merge_alg = merge_dict[self.config.merge_alg]
                merge_num = self.config.merge_freq // self.config.save_freq
                assert len(self.memory_weight) >= merge_num
                new_merge_weight = merge_alg.execute([self.config.weights / merge_num for _ in range(merge_num)], self.original_layer.weight, self.memory_weight[-merge_num:], densities=self.config.densities)
                min_a = 1e9
                for _ in range(merge_num):
                    self.memory_weight.pop()
                    edit_act = self.memory_mean_act.pop()
                    min_a = min(min_a, edit_act.min_act())
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                self.memory_weight.append(new_merge_weight)
                self.memory_mean_act.append(EditingMeanAct(min_a=min_a))
                print(len(self.memory_weight))
                assert len(self.memory_mean_act) == len(self.memory_weight)
                self.merge_cnt += 1
        else:
            merge_alg = merge_dict[self.config.merge_alg]
            cur_new_weight = merge_alg.execute(0.5, self.layer.weight, [self.new_weight],
                                               densities=self.config.densities)
            self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
            self.new_weight = copy.deepcopy(self.original_layer.weight)

    def merge_moe_weight(self):
        merge_alg = merge_dict[self.config.merge_alg]
        if self.original_layer.weight.equal(self.layer.weight):
            cur_new_weight = merge_alg.execute([self.config.weights / len(self.moe_weight) for _ in range(len(self.moe_weight))], self.original_layer.weight, self.moe_weight, densities=self.config.densities)
        else:
            cur_new_weight = merge_alg.execute([0.4 / len(self.moe_weight) for _ in range(len(self.moe_weight))] + [0.6], self.original_layer.weight, self.moe_weight + [self.layer.weight], densities=self.config.densities)
        # self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
        self.new_weight = copy.deepcopy(self.original_layer.weight)

        merge_num = len(self.moe_weight)
        min_a = 1e9
        for _ in range(merge_num):
            self.moe_weight.pop()
            edit_act = self.moe_mean_act.pop()
            min_a = min(min_a, edit_act.min_act())
        self.moe_weight.append(cur_new_weight)
        self.moe_mean_act.append(EditingMeanAct(min_a=min_a))
        print(len(self.moe_weight))
        assert len(self.moe_mean_act) == len(self.moe_weight)
        self.merge_cnt += 1

        self.merge_cluster_center()

    def merge_cluster_center(self):
        all_clusters = self.moe_cluster
        total_samples = sum(len(cluster['sample_id']) for cluster in all_clusters)

        # 梯度加权求和（梯度矩阵 * 对应样本数）
        weighted_grad = torch.zeros_like(all_clusters[0]['grad_matrix'])
        for cluster in all_clusters:
            weighted_grad += cluster['grad_matrix'] * len(cluster['sample_id'])

        # 计算平均梯度
        merged_grad = weighted_grad / total_samples

        # 合并所有mask（逻辑或操作）
        merged_mask = all_clusters[0]['mask'].clone()
        for cluster in all_clusters[1:]:
            merged_mask = torch.logical_or(merged_mask, cluster['mask'])

        # 合并所有sample_id（保留顺序）
        merged_samples = []
        for cluster in all_clusters:
            merged_samples.extend(cluster['sample_id'])

        # 生成新ID（假设使用最大ID+1的逻辑，或固定为0）
        new_cluster_id = 0

        # 清空原有聚类并存储新聚类
        self.moe_cluster.clear()
        self.moe_cluster.append(
            {
            'grad_matrix': merged_grad,
            'mask': merged_mask,
            'sample_id': merged_samples
            }
        )

    def merge_moe_weight_m(self):
        merge_alg = merge_dict[self.config.merge_alg]

        # 持续合并直到聚类数量不超过m
        while len(self.moe_cluster) > self.config.max_merge_num:
            # 准备梯度矩阵和掩码列表
            grad_list = [cluster['grad_matrix'] for cluster in self.moe_cluster]
            mask_list = [cluster['mask'] for cluster in self.moe_cluster]

            # 计算相似度矩阵
            sim_matrix = self.build_sim_matrix_loop(grad_list, mask_list)
            n = len(sim_matrix)

            # 寻找最大相似度的两个聚类
            max_sim = -1
            best_pair = (-1, -1)
            for i in range(n):
                for j in range(i+1, n):
                    if sim_matrix[i][j] > max_sim:
                        max_sim = sim_matrix[i][j]
                        best_pair = (i, j)

            if best_pair == (-1, -1):
                break  # 无有效合并对

            i, j = best_pair

            # 合并聚类中心
            cluster_i = self.moe_cluster[i]
            cluster_j = self.moe_cluster[j]
            merged_cluster = self._merge_cluster_group([cluster_i, cluster_j])

            # 计算合并权重（基于样本数）
            s_i = len(cluster_i['sample_id'])
            s_j = len(cluster_j['sample_id'])
            total = s_i + s_j
            # weights = [s_i / total, s_j / total]
            weights = [self.config.weights / 2, self.config.weights / 2]

            # 合并对应的MOE权重
            moe_weights = [self.moe_weight[i], self.moe_weight[j]]
            merged_weight = merge_alg.execute(
                weights,
                self.original_layer.weight,
                moe_weights,
                densities=self.config.densities
            )

            # 合并EditingMeanAct
            min_act = min(self.moe_mean_act[i].min_act(), self.moe_mean_act[j].min_act())
            new_mean_act = EditingMeanAct(min_a=min_act)

            # 移除旧条目（注意顺序）
            if i < j:
                self.moe_cluster.pop(j)
                self.moe_cluster.pop(i)
                self.moe_weight.pop(j)
                self.moe_weight.pop(i)
                self.moe_mean_act.pop(j)
                self.moe_mean_act.pop(i)
            else:
                self.moe_cluster.pop(i)
                self.moe_cluster.pop(j)
                self.moe_weight.pop(i)
                self.moe_weight.pop(j)
                self.moe_mean_act.pop(i)
                self.moe_mean_act.pop(j)

            # 添加新条目
            self.moe_cluster.append(merged_cluster)
            self.moe_weight.append(merged_weight)
            self.moe_mean_act.append(new_mean_act)

            self.merge_cnt += 1

        # 更新层的权重（根据需求调整）
        if len(self.moe_weight) == 1:
            self.layer.weight = torch.nn.Parameter(self.moe_weight[0].clone(), requires_grad=False)

        for i in range(len(self.moe_cluster)):
            print("adapter",i,"samples:",self.moe_cluster[i]['sample_id'], flush=True)

    def _merge_cluster_group(self, clusters):
        """合并一组聚类中心"""
        total_samples = sum(len(c['sample_id']) for c in clusters)

        # 梯度加权平均
        # weighted_grad = sum(c['grad_matrix'] * len(c['sample_id']) for c in clusters)
        weighted_grad = sum(c['grad_matrix'] for c in clusters)
        merged_grad = weighted_grad / total_samples

        # 掩码并集
        merged_mask = clusters[0]['mask'].clone()
        for c in clusters[1:]:
            merged_mask = torch.logical_or(merged_mask, c['mask'])

        # 合并样本ID
        merged_samples = []
        for c in clusters:
            merged_samples.extend(c['sample_id'])

        return {
            'grad_matrix': merged_grad,
            'mask': merged_mask,
            'sample_id': merged_samples
        }

    def merge_moe_weight_pair(self):
        merge_alg = merge_dict[self.config.merge_alg]

        while len(self.moe_cluster) > self.config.max_merge_num:
            current_n = len(self.moe_cluster)
            need = current_n - self.config.max_merge_num
            k = min(current_n // 2, need)
            if k <= 0:
                break

            # 计算相似度矩阵
            grad_list = [c['grad_matrix'] for c in self.moe_cluster]
            mask_list = [c['mask'] for c in self.moe_cluster]
            sim_matrix = self.build_sim_matrix_loop(grad_list, mask_list)

            # 生成所有候选对并按相似度排序
            candidate_pairs = []
            for i in range(current_n):
                for j in range(i+1, current_n):
                    candidate_pairs.append( (i, j, sim_matrix[i][j]) )
            candidate_pairs.sort(key=lambda x: -x[2])

            # 选择不重叠的top k对
            selected_pairs = []
            used = set()
            for pair in candidate_pairs:
                i, j, _ = pair
                if i not in used and j not in used and len(selected_pairs) < k:
                    selected_pairs.append( (i, j) )
                    used.update({i, j})

            if not selected_pairs:
                break

            # 按索引降序处理避免冲突
            selected_pairs = sorted(selected_pairs, key=lambda x: -max(x[0], x[1]))

            # 批量合并并缓存结果
            merge_buffer = []
            for i, j in selected_pairs:
                # 合并聚类中心
                cluster_i = self.moe_cluster[i]
                cluster_j = self.moe_cluster[j]
                merged_cluster = self._merge_cluster_group([cluster_i, cluster_j])

                # 合并权重
                s_i = len(cluster_i['sample_id'])
                s_j = len(cluster_j['sample_id'])
                weights = [s_i/(s_i+s_j), s_j/(s_i+s_j)]
                merged_weight = merge_alg.execute(
                    weights,
                    self.original_layer.weight,
                    [self.moe_weight[i], self.moe_weight[j]],
                    densities=self.config.densities
                )

                # 合并激活记录
                min_act = min(self.moe_mean_act[i].min_act(),
                            self.moe_mean_act[j].min_act())

                merge_buffer.append( (i, j, merged_cluster, merged_weight,
                                    EditingMeanAct(min_a=min_act)) )

            # 执行实际删除和添加操作
            for i, j, cluster, weight, act in merge_buffer:
                # 先删除大的索引
                idxs = sorted([i, j], reverse=True)
                for idx in idxs:
                    del self.moe_cluster[idx]
                    del self.moe_weight[idx]
                    del self.moe_mean_act[idx]
                # 添加新元素
                self.moe_cluster.append(cluster)
                self.moe_weight.append(weight)
                self.moe_mean_act.append(act)

            self.merge_cnt += len(selected_pairs)

        # 最终权重处理
        if len(self.moe_weight) == 1:
            self.layer.weight = torch.nn.Parameter(self.moe_weight[0].clone(),
                                                requires_grad=False)

    # def _merge_cluster_pair(self, c1, c2):
    #     """合并两个聚类中心的工具函数"""
    #     total = len(c1['sample_id']) + len(c2['sample_id'])
    #     return {
    #         'grad_matrix': (c1['grad_matrix'] * len(c1['sample_id']) +
    #                     c2['grad_matrix'] * len(c2['sample_id'])) / total,
    #         'mask': torch.logical_or(c1['mask'], c2['mask']),
    #         'sample_id': c1['sample_id'] + c2['sample_id']
    #     }

    def save_editing_activation(self):
        in_scope_dist = euc(self.original_layer_output[:-1, ...], self.new_weight_layer_output[:-1, ...], self.config)
        self.editing_mean_act.update(in_scope_dist.mean().item())

    def generate_mask_sensegrad(self, mask):
        if self.config.grad_sensetive_area and not self.config.rand_mask:
            self.weight_mask = mask
        else:
            # mask_new = torch.logical_and(mask, cluster['mask'])
            p_grad_lora = self.lora_weight.B.reshape(-1)
            p_mask = np.random.choice([1, 0], size=p_grad_lora.size()[0], p=[1-self.config.init_train_ratio, self.config.init_train_ratio])
            p_mask = torch.from_numpy(p_mask).to(p_grad_lora.device)
            self.weight_mask = p_mask

        p_grad_lora_A = self.lora_weight.A.reshape(-1)
        if self.config.use_gamma_para:
            p_mask_A = np.random.choice([1, 0], size=p_grad_lora_A.size()[0], p=[1-self.config.ga_para_1 / 10, self.config.ga_para_1 / 10])
        else:
            p_mask_A = np.random.choice([1, 0], size=p_grad_lora_A.size()[0], p=[1-self.config.init_train_ratio, self.config.init_train_ratio])

        p_mask_A = torch.from_numpy(p_mask_A).to(p_grad_lora_A.device)
        self.weight_mask_A = p_mask_A

        # self.weight_mask = mask

    def generate_activation_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        p_mask = np.random.choice([1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio])
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask

    def generate_non_overlapping_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        mask_size = int(mask_ratio * p_grad.size()[0])
        if self.used_mask is None:
            self.used_mask = np.zeros(p_grad.size()[0], dtype=bool)
        available_indices = np.where(~self.used_mask)[0]  # 获取未被遮罩的元素索引
        if len(available_indices) < mask_size:
            raise ValueError("Not enough unused elements to generate a new mask.")
        chosen_indices = np.random.choice(available_indices, size=mask_size, replace=False)
        mask_array = np.zeros(p_grad.size()[0], dtype=int)
        mask_array[chosen_indices] = 1
        self.used_mask[chosen_indices] = True  # 更新遮罩状态
        self.weight_mask = torch.from_numpy(mask_array).to(p_grad.device)

    def new_weight_forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.new_weight) if self.bias is None else torch.addmm(self.bias, input.view(-1, input.size(-1)), self.new_weight).view(input.size()[:-1] + (self.layer.nf,))

    def mask_new_weight_gradient(self):
        assert self.new_weight.grad is not None, print('Gradient Collection for New Weight error, gradient not found')
        # Add gradient mask after the loss updates
        p_size = self.new_weight.grad.size()
        p_grad = self.new_weight.grad.reshape(-1)

        # mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[.1, .9])).cuda()
        p_grad = p_grad * self.weight_mask
        self.new_weight.grad = p_grad.view(p_size).to(self.new_weight.grad.dtype)

    def mask_moe_weight_gradient(self):
        assert self.moe_weight[self.select_weight_id].B.grad is not None, print('Gradient Collection for New Weight error, gradient not found')
        # Add gradient mask after the loss updates
        p_size = self.moe_weight[self.select_weight_id].B.grad.size()
        p_grad = self.moe_weight[self.select_weight_id].B.grad.reshape(-1)

        # mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[.1, .9])).cuda()
        # if self.history_grad is not None:
        #     grad_times = p_grad.max().max() / self.history_grad.max().max()
        #     # p_grad += grad_times*self.history_grad
        #     dot_product = torch.dot(p_grad.flatten(), self.history_grad.flatten())
        #     norm_sq = torch.norm(self.history_grad) ** 2
        #     p_grad = p_grad - 0.1*grad_times* (dot_product / norm_sq) * self.history_grad

        p_grad = p_grad * self.weight_mask
        self.moe_weight[self.select_weight_id].B.grad = p_grad.view(p_size).to(self.moe_weight[self.select_weight_id].B.grad.dtype)

        p_size_A = self.moe_weight[self.select_weight_id].A.grad.size()
        p_grad_A = self.moe_weight[self.select_weight_id].A.grad.reshape(-1)
        p_grad_A = p_grad_A * self.weight_mask_A
        self.moe_weight[self.select_weight_id].A.grad = p_grad_A.view(p_size_A).to(self.moe_weight[self.select_weight_id].A.grad.dtype)

    def select_weight(self, sample_id, sample_grad, sample_mask, grad_A = None):
        max_sim = torch.tensor(0)
        min_cfl = torch.tensor(1)
        select_weight_id = 0
        create_new_weight_flag = True
        for i in range(len(self.moe_cluster)):
            cluster = self.moe_cluster[i]
            # union_mask = torch.logical_or(sample_mask, cluster['mask'])

            # conflict_mask_area = torch.logical_or(sample_mask, cluster['mask'])
            # conflict_ratio = torch.sum(conflict_mask_area)/conflict_mask_area.size(0)

            # masked_i = sample_grad * conflict_mask_area
            # masked_j = cluster['grad_matrix'] * conflict_mask_area
            if self.config.grad_sensetive_area:
                masked_i = sample_grad * cluster['mask']
            else:
                masked_i = sample_grad
            if self.config.new_lora_dir:
                masked_j = (self.lora_weight.B - self.moe_weight[i].B).reshape(-1)
            else:
                masked_j = cluster['grad_matrix'] * cluster['mask']

            # dot_product = torch.dot(masked_i, masked_j)
            # norm_product = torch.norm(masked_i) * torch.norm(masked_j)
            dot_product = torch.cosine_similarity(masked_i, masked_j, dim=0)

            if self.config.grad_sensetive_area:
                masked_i_A = grad_A * cluster['mask_A']
            else:
                masked_i_A = grad_A

            if self.config.new_lora_dir:
                masked_j_A = (self.lora_weight.A - self.moe_weight[i].A).reshape(-1)
            else:
                masked_j_A = cluster['grad_A'] * cluster['mask_A']
            dot_product_A = torch.cosine_similarity(masked_i_A, masked_j_A, dim=0)

            if self.config.no_A_in_produ:
                pass
            else:
                dot_product = (dot_product + dot_product_A)/2

            ratio = 1*len(cluster['sample_id'])
            dot_product = dot_product / ratio

            # if conflict_ratio < min_cfl and conflict_ratio < self.cfl_threshold:
            #     min_cfl = conflict_ratio
            #     select_weight_id = i
            #     create_new_weight_flag = False
            if self.config.grad_samples_select:
                if dot_product > max_sim and dot_product > self.sim_threshold and dot_product_A > 0 and len(cluster['sample_id'])<self.config.max_sample_num_per_cluster:
                    max_sim = dot_product
                    select_weight_id = i
                    create_new_weight_flag = False
            elif len(cluster['sample_id'])<self.config.max_sample_num_per_cluster:
                    max_sim = dot_product
                    select_weight_id = i
                    create_new_weight_flag = False
                    break

        if create_new_weight_flag:
            # self.weight_mask = cluster['mask']
            self.generate_mask_sensegrad(sample_mask)

            self.moe_weight.append(copy.deepcopy(self.lora_weight))
            cluster_dict = {'sample_id': [sample_id], 'grad_matrix':sample_grad,
                            'mask': self.weight_mask, 'mask_A':self.weight_mask_A, 'grad_A':grad_A}
            self.moe_cluster.append(cluster_dict)
            select_weight_id = len(self.moe_weight)-1
            self.moe_mean_act.append(EditingMeanAct())
            # self.cluster_activation.append(0)
            self.cluster_activation.append([])
            self.cluster_bound.append([])
            self.cluster_rel_activation.append([])
            self.cluster_rel_bound.append([])
            self.history_grad = None
        else:
            # overlap_mask = torch.logical_and(self.moe_cluster[select_weight_id]['mask'],sample_mask)
            # mask_new = torch.logical_xor(sample_mask, overlap_mask)
            # self.weight_mask = mask_new
            if self.config.grad_sensetive_area:
                self.weight_mask = sample_mask
            else:
                self.weight_mask = cluster['mask']

            self.weight_mask_A = cluster['mask_A']
            self.update_cluster_center(select_weight_id, sample_id, sample_grad, self.weight_mask, mask_A=self.weight_mask_A, grad_A=grad_A)
            self.history_grad = self.moe_cluster[select_weight_id]['history_grad']

            # self.ori_rel_outs = []
            if len(self.cluster_rel_activation[select_weight_id])>0:
                regu_act = torch.stack(self.cluster_rel_activation[select_weight_id])
                lora_i = self.moe_weight[select_weight_id]
                self.ori_rel_outs = lora_i(regu_act).detach()
            # for regu_act in self.cluster_rel_activation[select_weight_id]:
            #     lora_i = self.moe_weight[select_weight_id]
            #     rel_out = lora_i(regu_act)
            #     self.ori_rel_outs.append(rel_out.detach())

        # self.new_weight = self.moe_weight[select_weight]
        self.select_weight_id = select_weight_id
        self.set_moe_parameter_tunable()
        self.create_new_weight_flag = create_new_weight_flag

        self.moe_grad = True

        self.logs_dist = []
        self.logs_bound = []


    def update_cluster_center(self, seleted_id, sample_id, grad, grad_mask, mask_A=None, grad_A=None):
        num_edited = len(self.moe_cluster[seleted_id]['sample_id'])
        self.moe_cluster[seleted_id]['grad_matrix'] = (self.moe_cluster[seleted_id]['grad_matrix']*num_edited \
            + grad)/(num_edited+1)
        self.moe_cluster[seleted_id]['mask'] = torch.logical_or(self.moe_cluster[seleted_id]['mask'], grad_mask)
        self.moe_cluster[seleted_id]['sample_id'].append(sample_id)
        if grad_A is not None:
            self.moe_cluster[seleted_id]['grad_A'] = (self.moe_cluster[seleted_id]['grad_A']*num_edited \
            + grad_A)/(num_edited+1)
            # self.moe_cluster[seleted_id]['mask_A'] = torch.logical_or(self.moe_cluster[seleted_id]['mask'], grad_mask)



    def save_moe_activation(self):
        in_scope_dist = euc(self.original_layer_output[:-1, ...], self.moe_weight_layer_output[:-1, ...], self.config)
        self.moe_mean_act[self.select_weight_id].update(in_scope_dist.mean().item())

    def set_moe_parameter_tunable(self):
        for idx, lora_layer in enumerate(self.moe_weight):
            lora_layer.B.requires_grad = (idx == self.select_weight_id)
            lora_layer.A.requires_grad = (idx == self.select_weight_id)
        self.new_weight.requires_grad = False
        self.lora_weight.A.requires_grad = False
        self.lora_weight.B.requires_grad = False

    def set_lora_parameter_tunable(self):
        self.lora_weight.A.requires_grad = True
        self.lora_weight.B.requires_grad = True


    def update_cluster_activation(self, sample_act):
        num_edited = len(self.moe_cluster[self.select_weight_id]['sample_id'])
        self.cluster_activation[self.select_weight_id] = self.cluster_activation[self.select_weight_id] * num_edited + torch.mean(sample_act,dim=0)

    def forward(self, *args):
        if self.editing:
            if not self.moe_grad:
                lora_out = self.lora_weight(*args)
                self.original_layer_output = self.original_layer(*args)
                layer_out = self.original_layer_output + lora_out
                self.new_weight_layer_output = layer_out
            else:
                selected_moe_weight= self.moe_weight[self.select_weight_id]
                self.original_layer_output = self.original_layer(*args)

                lora_effect = selected_moe_weight(*args)
                layer_out = self.original_layer_output + lora_effect
                self.moe_weight_layer_output = layer_out

                self.knowledge_act = args[0]
        else:
            print('---------------------------------------------------------------------', flush=True)
            if self.config.sensemask:
                # if not self.config.merge_moe:
                original_layer_output = self.original_layer(*args)
                min_dist = 0
                layer_out = original_layer_output

                # lora_out = []
                # for lora_layer in self.moe_weight:
                #     lora_effect = lora_layer(*args)
                #     layer_out += lora_effect

                select_moe_id = []
                # select_moe_id  = -1
                select_act_dist = []
                if self.config.select_all:
                    select_moe_id = list(range(len(self.cluster_activation)))
                elif self.config.max_sample_num_per_cluster >= 100:
                    select_moe_id = [0]
                elif self.config.select_logic_1: # rel dist 找到相似度最大的（all_sample），判断是否在区域内
                    max_act_dist = 0, -1, -1 # dist, cluster id, sample_id
                    for cluster_act_idx in range(len(self.cluster_activation)):
                        for a_index, (act, rel_act) in enumerate(zip(self.cluster_activation[cluster_act_idx], self.cluster_rel_activation[cluster_act_idx])):
                            # act -- subject 的 token 拿出来
                            # rel_act -- prompt 的 token
                            if self.config.use_chat_template:
                                act_dist = torch.cosine_similarity(rel_act, torch.mean(args[0].squeeze(0),dim=0), dim=0).item(), cluster_act_idx, a_index
                            else:
                                act_dist = torch.cosine_similarity(act, torch.mean(args[0].squeeze(0),dim=0), dim=0).item(), cluster_act_idx, a_index


                            if act_dist > max_act_dist:
                                max_act_dist = act_dist

                    act_dist, cluster_act_idx, a_index = max_act_dist
                    act_bound = self.cluster_bound[cluster_act_idx][a_index].item()

                    if cluster_act_idx > -1 and act_dist > act_bound: # 0.53 < 0.58
                        select_moe_id.append(cluster_act_idx)
                elif self.config.select_logic_A800:
                    for cluster_act_idx in range(len(self.cluster_activation)):
                        max_act_dist = 0
                        max_rel_act_dist = 0

                        act_bound = 0
                        for a_index, (act, rel_act) in enumerate(zip(self.cluster_activation[cluster_act_idx], self.cluster_rel_activation[cluster_act_idx])):
                            act_dist = torch.cosine_similarity(act, torch.mean(args[0].squeeze(0),dim=0), dim=0)
                            rel_act_dist = torch.cosine_similarity(rel_act, torch.mean(args[0].squeeze(0),dim=0), dim=0)
                            # act_dist = torch.cosine_similarity(act, args[0].squeeze(0)[-1], dim=0)
                            # rel_act_dist = torch.cosine_similarity(rel_act, args[0].squeeze(0)[-1], dim=0)
                        # act = self.cluster_activation[cluster_act_idx]
                        # act_dist = torch.cosine_similarity(act,torch.mean(args[0].squeeze(0),dim=0), dim=0)
                            use_dist = rel_act_dist if self.config.use_chat_template else act_dist



                            if (use_dist > max_act_dist).any(): # max_act_dist:
                                max_act_dist = use_dist

                                act_bound = self.cluster_bound[cluster_act_idx][a_index]

                                # rel_bound = self.cluster_rel_bound[cluster_act_idx][a_index]
                                # select_moe_id = cluster_act_idx
                        ### 选大于阈值的2个

                        num_lora = 1
                        if isinstance((max_act_dist > act_bound),Tensor):
                            bound_flag = (max_act_dist > act_bound).any()
                        else:
                            bound_flag = max_act_dist > act_bound

                        if bound_flag and len(select_moe_id) < num_lora:
                            select_moe_id.append(cluster_act_idx)
                            select_act_dist.append(max_act_dist)
                        elif len(select_moe_id) >= num_lora and max_act_dist > min(select_act_dist):
                            select_moe_id.pop(select_act_dist.index(min(select_act_dist)))
                            select_moe_id.append(cluster_act_idx)
                elif self.config.select_logic_new0507:
                    for cluster_act_idx in range(len(self.cluster_activation)):
                        max_act_dist = 0
                        max_rel_act_dist = 0

                        act_bound = 0
                        for a_index, act in enumerate(self.cluster_activation[cluster_act_idx]):
                            if self.config.act_key_last:
                                act_dist = torch.cosine_similarity(act, args[0].squeeze(0)[-1], dim=0)
                            else:
                                act_dist = torch.cosine_similarity(act, torch.mean(args[0].squeeze(0),dim=0), dim=0)


                            # rel_act_dist = torch.cosine_similarity(rel_act, torch.mean(args[0].squeeze(0),dim=0), dim=0)
                            # act_dist = torch.cosine_similarity(act, args[0].squeeze(0)[-1], dim=0)
                            # rel_act_dist = torch.cosine_similarity(rel_act, args[0].squeeze(0)[-1], dim=0)
                        # act = self.cluster_activation[cluster_act_idx]
                        # act_dist = torch.cosine_similarity(act,torch.mean(args[0].squeeze(0),dim=0), dim=0)
                            use_dist =  act_dist

                            act_bound = self.cluster_bound[cluster_act_idx][a_index]
                            num_lora = 1
                            if isinstance((use_dist > act_bound),Tensor):
                                bound_flag = (use_dist > act_bound).any()
                            else:
                                bound_flag = use_dist > act_bound

                            if bound_flag and len(select_moe_id) < num_lora:
                                select_moe_id.append(cluster_act_idx)
                                select_act_dist.append(use_dist)
                            elif bound_flag and len(select_moe_id) >= num_lora and use_dist > min(select_act_dist):
                                select_moe_id.pop(select_act_dist.index(min(select_act_dist)))
                                select_moe_id.append(cluster_act_idx)
                else:
                    # for cluster_act_idx in range(len(self.cluster_activation)):
                    #     max_act_dist = 0
                    #     max_rel_act_dist = 0

                    #     act_bound = 0
                    #     bound_flag = False
                    #     for a_index, (act, rel_act) in enumerate(zip(self.cluster_activation[cluster_act_idx], self.cluster_rel_activation[cluster_act_idx])):
                    #         act_dist = torch.cosine_similarity(act, torch.mean(args[0].squeeze(0),dim=0), dim=0)
                    #         rel_act_dist = torch.cosine_similarity(rel_act, torch.mean(args[0].squeeze(0),dim=0), dim=0)
                    #         # act_dist = torch.cosine_similarity(act, args[0].squeeze(0)[-1], dim=0)
                    #         # rel_act_dist = torch.cosine_similarity(rel_act, args[0].squeeze(0)[-1], dim=0)
                    #     # act = self.cluster_activation[cluster_act_idx]
                    #     # act_dist = torch.cosine_similarity(act,torch.mean(args[0].squeeze(0),dim=0), dim=0)

                    #         if (act_dist > max_act_dist).any(): # max_act_dist:
                    #             max_act_dist = act_dist
                    #             max_rel_act_dist = rel_act_dist

                    #         act_bound = self.cluster_bound[cluster_act_idx][a_index]
                    #         rel_bound = self.cluster_rel_bound[cluster_act_idx][a_index]
                    #             # select_moe_id = cluster_act_idx

                    #         self.logs_dist.append(act_dist)
                    #         self.logs_bound.append(act_bound)

                    #         if isinstance((act_dist > act_bound),Tensor):
                    #             if (act_dist > act_bound).any() and (act_dist > 0.3).any() :
                    #                 bound_flag = True
                    #         elif act_dist > act_bound and act_dist > 0.3:
                    #             bound_flag = True

                    #         if bound_flag:
                    #             print(act_bound, rel_bound, flush=True)
                    #     ### 选大于阈值的2个
                    #     num_lora = 1
                    #     # if isinstance((max_act_dist > act_bound),Tensor):
                    #     #     bound_flag = (max_act_dist > act_bound).any()
                    #     # else:
                    #     #     bound_flag = max_act_dist > act_bound

                    #     if bound_flag and len(select_moe_id) < num_lora:
                    #         select_moe_id.append(cluster_act_idx)
                    #         select_act_dist.append(max_act_dist)
                    #     elif bound_flag and len(select_moe_id) >= num_lora and max_act_dist > min(select_act_dist):
                    #         select_moe_id.pop(select_act_dist.index(min(select_act_dist)))
                    #         select_moe_id.append(cluster_act_idx)

                    #         select_act_dist.pop(select_act_dist.index(min(select_act_dist)))
                    #         select_act_dist.append(max_act_dist)
                    for cluster_act_idx in range(len(self.cluster_activation)):
                        max_act_dist = 0
                        max_rel_act_dist = 0

                        act_bound = 0
                        for a_index, (act, rel_act) in enumerate(zip(self.cluster_activation[cluster_act_idx], self.cluster_rel_activation[cluster_act_idx])):
                            act_dist = torch.cosine_similarity(act, torch.mean(args[0].squeeze(0),dim=0), dim=0)
                            rel_act_dist = torch.cosine_similarity(rel_act, torch.mean(args[0].squeeze(0),dim=0), dim=0)
                            # act_dist = torch.cosine_similarity(act, args[0].squeeze(0)[-1], dim=0)
                            # rel_act_dist = torch.cosine_similarity(rel_act, args[0].squeeze(0)[-1], dim=0)
                        # act = self.cluster_activation[cluster_act_idx]
                        # act_dist = torch.cosine_similarity(act,torch.mean(args[0].squeeze(0),dim=0), dim=0)
                            if (act_dist > max_act_dist).any(): # max_act_dist:
                                max_act_dist = act_dist
                                max_rel_act_dist = rel_act_dist

                                act_bound = self.cluster_bound[cluster_act_idx][a_index]
                                rel_bound = self.cluster_rel_bound[cluster_act_idx][a_index]
                                # select_moe_id = cluster_act_idx
                        ### 选大于阈值的2个
                        num_lora = 1
                        if isinstance((max_act_dist > act_bound),Tensor):
                            bound_flag = (max_act_dist > act_bound).any()
                        else:
                            bound_flag = max_act_dist > act_bound

                        if bound_flag and len(select_moe_id) < num_lora:
                            select_moe_id.append(cluster_act_idx)
                            select_act_dist.append(max_act_dist)
                        elif len(select_moe_id) >= num_lora and max_act_dist > min(select_act_dist):
                            select_moe_id.pop(select_act_dist.index(min(select_act_dist)))
                            select_moe_id.append(cluster_act_idx)

                for select_index in select_moe_id:
                    selected_moe_weight = self.moe_weight[select_index]
                    # layer_out = F.linear(*args, moe_weight)
                    lora_effect = selected_moe_weight(*args)
                    layer_out = layer_out + lora_effect

                ### 选相似度最大
                # if max_act_dist > 0.5:
                #     # select_moe_id.append(cluster_act_idx)
                #     selected_moe_weight = self.moe_weight[select_moe_id]
                #     # layer_out = F.linear(*args, moe_weight)
                #     lora_effect = selected_moe_weight(*args)
                #     layer_out = layer_out + lora_effect

                # if max_act_dist > 0.5:
                #     selected_moe_weight = self.moe_weight[select_moe_id]
                #     # layer_out = F.linear(*args, moe_weight)
                #     lora_effect = selected_moe_weight(*args)
                #     layer_out = original_layer_output + lora_effect
                # else:
                #     select_moe_id = -1

                    ###################
                # for i in range(len(self.moe_weight)):
                #     selected_moe_weight = self.moe_weight[i]
                #     # moe_weight_layer_output = F.linear(*args, moe_weight)
                #     lora_effect = selected_moe_weight(*args)
                #     moe_weight_layer_output = original_layer_output + lora_effect

                #     dist = euc(original_layer_output, moe_weight_layer_output, self.config, infer=True)
                #     if dist > min_dist and dist > self.moe_mean_act[i].min_act() * self.config.act_ratio:
                #         layer_out = moe_weight_layer_output
                #         min_dist = dist
                #         select_moe_id = i
                print(select_moe_id, flush=True)
                # else:
                # original_layer_output = self.original_layer(*args)
                # layer_output = self.layer(*args)
                # new_weight_layer_output = self.new_weight_forward(*args)
                # dist2 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                # dist1 = euc(original_layer_output, layer_output, self.config, infer=True)
                # threshold = self.moe_mean_act[0].min_act() * self.config.act_ratio

                # if dist1.item() < threshold and dist2.item() < threshold:
                #     layer_out = original_layer_output
                # elif dist1.item() > dist2.item():
                #     layer_out = layer_output
                # else:
                #     layer_out = new_weight_layer_output
            else:
                if not self.config.retrieve:
                    original_layer_output = self.original_layer(*args)
                    layer_output = self.layer(*args)
                    new_weight_layer_output = self.new_weight_forward(*args)
                    dist2 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                    dist1 = euc(original_layer_output, layer_output, self.config, infer=True)
                    threshold = self.editing_mean_act.min_act() * self.config.act_ratio
            #cancel next 5 line
                    if dist1.item() < threshold and dist2.item() < threshold:
                        layer_out = original_layer_output
                    elif dist1.item() > dist2.item():
                        layer_out = layer_output
                    else:
                        layer_out = new_weight_layer_output
                else:
                    original_layer_output = self.original_layer(*args)
                    new_weight_layer_output = self.new_weight_forward(*args)
                    dist1 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                    threshold = self.editing_mean_act.min_act() * self.config.act_ratio
                    min_dist = dist1
                    if min_dist.item() < threshold:
                        layer_out = original_layer_output
                    else:
                        layer_out = new_weight_layer_output

                    for i in range(len(self.memory_weight)):
                        memory_retrieve_weight = self.memory_weight[i]
                        memory_weight_layer_output = F.linear(*args, memory_retrieve_weight)
                        dist = euc(original_layer_output, memory_weight_layer_output, self.config, infer=True)
                        if dist > min_dist and dist > self.memory_mean_act[i].min_act() * self.config.act_ratio:
                            layer_out = memory_weight_layer_output
                            min_dist = dist
        return layer_out


    def build_sim_matrix_loop(self, gradient_vectors, fisher_masks):
        """内存优化的循环计算版本（支持列表输入）"""
        # 将输入统一转换为列表形式，避免显存占用
        if isinstance(gradient_vectors, torch.Tensor):
            grad_list = [gradient_vectors[i] for i in range(gradient_vectors.size(0))]
        else:
            grad_list = gradient_vectors

        n = len(grad_list)
        device = grad_list[0].device

        # 同样处理fisher_masks
        if isinstance(fisher_masks, torch.Tensor):
            fisher_list = [fisher_masks[i] for i in range(n)]
        else:
            fisher_list = fisher_masks

        assert len(fisher_list) == n, "gradient_vectors和fisher_masks长度不一致"

        sim_matrix = torch.zeros((n, n), device=device)
        eps = 1e-8

        with torch.no_grad():
            for i in range(n):
                grad_i = grad_list[i]
                mask_i = fisher_list[i]

                for j in range(i, n):
                    grad_j = grad_list[j]
                    mask_j = fisher_list[j]

                    # 计算并集mask
                    union_mask = torch.logical_or(mask_i, mask_j)

                    # 向量化计算
                    masked_i = grad_i * union_mask
                    masked_j = grad_j * union_mask

                    # 计算相似度
                    dot_product = torch.dot(masked_i, masked_j)
                    norm_product = torch.norm(masked_i) * torch.norm(masked_j)

                    # 对称赋值
                    if norm_product > eps:
                        sim_val = dot_product / norm_product
                        sim_matrix[i, j] = sim_val
                        sim_matrix[j, i] = sim_val
                    else:
                        sim_matrix[i, j] = 0.0
                        sim_matrix[j, i] = 0.0

                    # 及时释放中间变量
                    del union_mask, masked_i, masked_j, dot_product, norm_product

                # 定期清空缓存
                if i % 100 == 0:
                    torch.cuda.empty_cache()

        return sim_matrix


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha=1.0):
        """
        独立LoRA层实现
        :param in_dim: 原始层输入维度（对应down_proj的4096）
        :param out_dim: 原始层输出维度（对应down_proj的11008）
        :param rank: 低秩矩阵的维度
        :param alpha: 缩放因子，默认合并到前向计算中
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 初始化参数矩阵
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

        # 使用Kaiming初始化（与原实现保持一致）
        # nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.normal_(self.B, mean=0.0, std=0.02)

    def forward(self, x):
        """
        输入x形状: [..., in_dim]
        输出形状: [..., out_dim]
        """
        # 计算低秩适应 (x @ A @ B) * alpha
        lora_output = (x @ self.A) @ self.B
        return lora_output * self.alpha