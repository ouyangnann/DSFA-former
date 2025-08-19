# -*- coding: utf-8 -*-
import os
import sys
import time
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

# 确保项目根路径在 sys.path，避免找不到 lib 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# 依赖你已有的项目结构
from lib import utils, metrics
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.iGraphformer import iGraphformer
from model.pytorch.tsf_model import TSFModel, count_parameters
from model.pytorch.gtn_model import GTNModel
from model.pytorch.itgtn import iTGTNModel
from model.pytorch.itn import iTNModel
from model.pytorch.itpgtn import iTPGTNModel
from model.pytorch.itgcn import iTGCNModel


class ModelsSupervisor:
    def __init__(self, models, pretrained_model_path, config_filename, cuda, **kwargs):
        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        # 覆盖 batch_size（只为可视化取样更方便，也可改回配置中的值）
        self.batch_size = 1
        self._data_kwargs['seq_len'] = int(self._model_kwargs.get('seq_len'))
        self._data_kwargs['horizon'] = int(self._model_kwargs.get('horizon'))

        # 日志与输出目录
        self._log_dir = self._get_log_dir(models, config_filename, kwargs)
        self.importance_vis_dir = os.path.join(self._log_dir, "node_importance")
        os.makedirs(self.importance_vis_dir, exist_ok=True)

        # 数据集名称
        self.dataset_name = self._data_kwargs.get('dataset', 'pems-bay').lower()
        print(f"[Init] Dataset name: {self.dataset_name}")

        self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # 加载数据
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        # 图
        graph_pkl_filename = self._data_kwargs.get('graph_pkl_filename')
        if self._data_kwargs.get('use_graph'):
            _, _, adj_mx = utils.load_graph_data(graph_pkl_filename)
            self.graph = adj_mx
        else:
            self.graph = None

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.horizon = int(self._model_kwargs.get('horizon', 1))

        model_list = {
            'dcrnn': DCRNNModel,
            'tsf': TSFModel,
            'gtn': GTNModel,
            'itn': iTNModel,
            'itgtn': iTGTNModel,
            'itgcn': iTGCNModel,
            'itpgtn': iTPGTNModel,
            'iGraphformer': iGraphformer
        }
        init_model = model_list[models]

        if models == 'iGraphformer':
            self.amodel = init_model(self._logger, self.graph, cuda, **self._model_kwargs).to(self.device)
        else:
            self.amodel = init_model(self._logger, cuda, **self._model_kwargs).to(self.device)

        self._logger.info("config_filename: %s", config_filename)
        self._logger.info("device: %s", self.device)
        self._logger.info("Model created")

        self.load_pre_model(pretrained_model_path)

        # 坐标文件检查
        print("\n[Check] Verifying coordinate files...")
        self._verify_coordinate_files()
        print("[Check] Coordinate verification complete.\n")
        
        self.topk=10

    # ---------------- 工具与加载 ----------------
    @staticmethod
    def _get_log_dir(loadmodel, config_name, kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            seq_len = int(kwargs['model'].get('seq_len'))
            horizon = int(kwargs['model'].get('horizon'))
            run_id = '%s_%s_l_%d_h_%d_lr_%g_bs_%d/' % (
                time.strftime('%Y%m%d_%H%M%S'),
                loadmodel,
                seq_len, horizon,
                learning_rate, batch_size
            )
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, 'log', loadmodel, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        shutil.copy(config_name, log_dir)
        return log_dir

    def load_pre_model(self, model_path):
        if not model_path:
            return
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        # 移除临时缓存参数键
        filtered_state_dict = {k: v for k, v in state_dict.items()
                               if 'last_W_q' not in k and 'last_weights' not in k}
        self.amodel.load_state_dict(filtered_state_dict, strict=False)
        self._logger.info("Loaded pretrained model from: %s", model_path)

    def _setup_graph(self):
        with torch.no_grad():
            self.amodel = self.amodel.eval()
            val_iterator = self._data['val_loader'].get_iterator()
            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                if isinstance(self.amodel, iGraphformer):
                    self.amodel(x, self.batch_size)
                else:
                    self.amodel(x, self.graph, batches_seen=0)
                break

    # ---------------- 评估入口 ----------------
    def evaluate(self, dataset='test'):
        print(f"[NodeImportance] Saving to: {self.importance_vis_dir}")
        with torch.no_grad():
            self.amodel = self.amodel.eval()
            data_loader = self._data[f'{dataset}_loader'].get_iterator()
            y_truths, y_preds, loss = [], [], []
            for idx, (x, y) in enumerate(data_loader):
                if idx % 12 != 0:  # 避免重叠
                    continue
                x, y = self._prepare_data(x, y)
                if isinstance(self.amodel, iGraphformer):
                    output = self.amodel(x)
                else:
                    output = self.amodel(x, self.graph)

                y_truth = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)

                loss.append(metrics.masked_mae_torch(y_pred, y_truth).item())
                y_truths.append(y_truth.cpu())
                y_preds.append(y_pred.cpu())

            y_preds = np.concatenate(y_preds, axis=1).reshape(-1, self.num_nodes)
            y_truths = np.concatenate(y_truths, axis=1).reshape(-1, self.num_nodes)
            print(f"[Eval] Average loss: {np.mean(loss):.4f}")

            # 评估节点重要性并可视化
            self.evaluate_node_importance(dataset)

        return y_truths, y_preds

   

    # ---- 基于中间变量的分数（可视化阶段1/2）----
    def _node_scores_from_Wq(self, W_q: torch.Tensor) -> np.ndarray:
        """
        基于压缩层权重 W_q 计算节点级分数。
        W_q: [C, N]  -> node_score[n] = mean_c |W_q[c, n]|
        返回 [N] ∈ [0,1]
        """
        if isinstance(W_q, torch.Tensor):
            W_q = W_q.detach().cpu().numpy()
        node_score = np.mean(np.abs(W_q), axis=0)  # [N]
        node_score = (node_score - node_score.min()) / (node_score.max() - node_score.min() + 1e-8)
        return node_score

    def _node_scores_from_weights_Wq(self, weights: torch.Tensor, W_q: torch.Tensor) -> np.ndarray:
        """
        基于阶段 softmax 权重 + W_q 的投影计算节点级分数。
        weights: [B, C, D]  -> avg over (B, D) -> [C]
        W_q: [C, N]
        返回 [N] ∈ [0,1]
        """
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        if isinstance(W_q, torch.Tensor):
            W_q = W_q.detach().cpu().numpy()
        w_c = weights.mean(axis=(0, 2))  # [C]
        node_score = np.abs(w_c @ W_q)   # [N]
        node_score = (node_score - node_score.min()) / (node_score.max() - node_score.min() + 1e-8)
        return node_score

    # ---------------- 坐标与列名工具 ----------------
    def _coords_file_for_dataset(self) -> str:
        """
        根据数据集名称返回默认坐标文件路径。
        - METR-LA:  data/sensor_graph/graph_sensor_locations.csv
        - PEMS-BAY: data/sensor_graph/graph_sensor_locations_bay.csv
        """
        sensor_graph_dir = os.path.join("data", "sensor_graph")
        if self.dataset_name in ["metr-la", "metr_la", "metrla", "la"]:
            return os.path.join(sensor_graph_dir, "graph_sensor_locations.csv")
        elif self.dataset_name in ["pems-bay", "pems_bay", "bay"]:
            return os.path.join(sensor_graph_dir, "graph_sensor_locations_bay.csv")
        # 兜底
        bay = os.path.join(sensor_graph_dir, "graph_sensor_locations_bay.csv")
        metr = os.path.join(sensor_graph_dir, "graph_sensor_locations.csv")
        return bay if os.path.exists(bay) else metr

    def _verify_coordinate_files(self):
        sensor_graph_dir = os.path.join("data", "sensor_graph")
        metr_path = os.path.join(sensor_graph_dir, "graph_sensor_locations.csv")
        bay_path = os.path.join(sensor_graph_dir, "graph_sensor_locations_bay.csv")

        def _check(p, name):
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    print(f"[Coords] {name} file OK: {p}, nodes={len(df)}")
                except Exception as e:
                    print(f"[Coords] {name} read error: {e}")
            else:
                print(f"[Coords] {name} file MISSING at {p}")

        _check(metr_path, "METR-LA")
        _check(bay_path, "PEMS-BAY")

    @staticmethod
    def _detect_lat_lon_columns(df: pd.DataFrame):
        """
        自动识别经纬度列名，返回 (lat_col, lon_col)
        """
        lat_col, lon_col = None, None
        for c in df.columns:
            lc = c.strip().lower()
            if lat_col is None and "lat" in lc:
                lat_col = c
            if lon_col is None and ("lon" in lc or "long" in lc):
                lon_col = c
        if lat_col is None or lon_col is None:
            raise ValueError(f"Cannot find latitude/longitude columns. Available: {df.columns.tolist()}")
        return lat_col, lon_col
    
    def visualize_node_importance(self, importance: np.ndarray, epoch: int, step: int,
                              tag: str = "overall"):
        """
        使用地理经纬度绘制节点重要性散点图（不绘制边/子图）。
        仅对 Top-K 节点标注节点索引号，文件名中包含 tag（如 last_weights_2）。
        """
        os.makedirs(self.importance_vis_dir, exist_ok=True)
        coords_file = self._coords_file_for_dataset()
        if not os.path.exists(coords_file):
            raise FileNotFoundError(f"Coordinate CSV not found: {coords_file}")

        df = pd.read_csv(coords_file)
        if set(df.columns) == {0, 1, 2}:
            df.columns = ['sensor_id', 'latitude', 'longitude']

        lat_col, lon_col = self._detect_lat_lon_columns(df)
        coords = df[[lon_col, lat_col]].values  # 经度在前、纬度在后

        if coords.shape[0] != self.num_nodes:
            self._logger.warning(
                f"[Coords] CSV nodes={coords.shape[0]} != model nodes={self.num_nodes}. "
                f"{'Truncating' if coords.shape[0] > self.num_nodes else 'Padding'} to match."
            )
            if coords.shape[0] > self.num_nodes:
                coords = coords[:self.num_nodes]
            else:
                pad = np.tile(coords[-1], (self.num_nodes - coords.shape[0], 1))
                coords = np.vstack([coords, pad])

        imp = np.asarray(importance).reshape(-1)
        imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)

        sizes = 20 + 180 * imp
        cmap = plt.cm.viridis

        plt.figure(figsize=(10, 8))
        sc = plt.scatter(coords[:, 0], coords[:, 1],
                        c=imp, cmap=cmap, s=sizes,
                        edgecolors='k', linewidths=0.25, alpha=0.95)

        cbar = plt.colorbar(sc)
        cbar.set_label("Node Importance (normalized)", fontsize=12)

        # 仅标注 Top-K 节点的“节点索引号”
        topk_num = min(5, len(imp))
        topk_idx = np.argsort(imp)[-topk_num:][::-1]   # Top-5
        bottomk_idx = np.argsort(imp)[:topk_num]       # Bottom-5

        ax = plt.gca()
        for idx in topk_idx:
            ax.annotate(str(idx),
                        (coords[idx, 0], coords[idx, 1]),
                        xytext=(5, 7), textcoords="offset points",
                        fontsize=8, fontweight="bold", color="black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.75))
        for idx in bottomk_idx:
            ax.annotate(str(idx),
                        (coords[idx, 0], coords[idx, 1]),
                        xytext=(5, -12), textcoords="offset points",
                        fontsize=8, fontweight="bold", color="red",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.75))

        plt.title(f"{self.dataset_name.upper()} Node Importance [{tag}]\nEpoch {epoch}, Step {step}", fontsize=12)
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.grid(alpha=0.25, linestyle='--')

        out_name = f"{self.dataset_name}_importance_{tag}_epoch{epoch}_step{step:05d}.png"
        out_path = os.path.join(self.importance_vis_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=400, bbox_inches='tight')
        plt.close()
        self._logger.info(f"[Geo Importance] Saved: {out_path}")

    def save_extreme_timeseries_from_raw(self,
                                     importance_scores,
                                     raw_speed,
                                     sample_idx,
                                     tag="last_weights_1",
                                     k=5,
                                     last_len=288*7,
                                     individual_figsize=(16, 3)):
        """
        使用原始数据绘制 Top-k 和 Bottom-k 的时间序列：
        - 为每个节点各自保存一张“扁长型”PNG
        - 生成两张“汇总栅格图”：Top 一页、Bottom 一页
        >>> 统一纵轴尺度：Top/Bottom/单图均使用同一对 y-lims
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm

        assert raw_speed.ndim == 2, "raw_speed must be (T, N)"
        T, N = raw_speed.shape
        k = min(k, len(importance_scores))

        # 排序并取索引
        order = np.argsort(importance_scores)   # 升序
        bottom_idx = order[:k]
        top_idx = order[-k:][::-1]

        # 取最后 last_len 步（若不合法则画全长）
        if last_len is None or last_len <= 0 or last_len > T:
            s, e = 0, T
        else:
            s, e = max(0, T - last_len), T
        ts = raw_speed[s:e, :]
        t = np.arange(ts.shape[0])

        # === 统一纵轴范围（Top/Bottom/单图一致）===
        sel_nodes = np.concatenate([top_idx, bottom_idx])
        vals = ts[:, sel_nodes]
        y_min = float(np.nanmin(vals))
        y_max = float(np.nanmax(vals))
        pad = 0.05 * (y_max - y_min + 1e-9)
        y_lo, y_hi = y_min - pad, y_max + pad

        cmap = cm.get_cmap('tab10')
        os.makedirs(self.importance_vis_dir, exist_ok=True)

        # -------- Top-k：每节点单图（扁长型 & 统一y轴）--------
        for node_idx in top_idx:
            plt.figure(figsize=individual_figsize)
            plt.plot(t, ts[:, node_idx], linewidth=2.0, color=cmap(0))
            plt.ylim(y_lo, y_hi)
            plt.title(f"Node {node_idx} (Top) from RAW [{tag}] Sample {sample_idx}")
            plt.xlabel("Time Index"); plt.ylabel("Speed")
            plt.grid(alpha=0.3, linestyle='--')
            single_top_path = os.path.join(
                self.importance_vis_dir,
                f"{self.dataset_name}_node{node_idx}_timeseries_top_{tag}_sample{sample_idx}.png"
            )
            plt.tight_layout(); plt.savefig(single_top_path, dpi=400, bbox_inches='tight')
            plt.close()
            self._logger.info(f"[Top Node RAW] Saved: {single_top_path}")

        # -------- Bottom-k：每节点单图（扁长型 & 统一y轴）--------
        for node_idx in bottom_idx:
            plt.figure(figsize=individual_figsize)
            plt.plot(t, ts[:, node_idx], linewidth=2.0, color=cmap(1))
            plt.ylim(y_lo, y_hi)
            plt.title(f"Node {node_idx} (Bottom) from RAW [{tag}] Sample {sample_idx}")
            plt.xlabel("Time Index"); plt.ylabel("Speed")
            plt.grid(alpha=0.3, linestyle='--')
            single_bottom_path = os.path.join(
                self.importance_vis_dir,
                f"{self.dataset_name}_node{node_idx}_timeseries_bottom_{tag}_sample{sample_idx}.png"
            )
            plt.tight_layout(); plt.savefig(single_bottom_path, dpi=400, bbox_inches='tight')
            plt.close()
            self._logger.info(f"[Bottom Node RAW] Saved: {single_bottom_path}")

        # -------- Top-k：栅格汇总（全部放到同一页 & 统一y轴）--------
        fig_h = max(2.2 * k, 6.0)             # 根据 k 调整整页高度
        fig_w = 18.0
        fig, axes = plt.subplots(nrows=k, ncols=1, figsize=(fig_w, fig_h), sharex=True)
        if k == 1:
            axes = [axes]
        for i, node_idx in enumerate(top_idx):
            ax = axes[i]
            ax.plot(t, ts[:, node_idx], linewidth=2.0, color=cmap(i % cmap.N))
            ax.set_ylim(y_lo, y_hi)
            ax.set_title(f"Top-{i+1}: Node {node_idx}")
            ax.grid(alpha=0.3, linestyle='--')
            if i == k - 1:
                ax.set_xlabel("Time Index")
            ax.set_ylabel("Speed")
        plt.tight_layout()
        top_grid_path = os.path.join(
            self.importance_vis_dir,
            f"{self.dataset_name}_timeseries_top_grid_{tag}_sample{sample_idx}.png"
        )
        fig.savefig(top_grid_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        self._logger.info(f"[Top Grid RAW] Saved: {top_grid_path}")

        # -------- Bottom-k：栅格汇总（全部放到同一页 & 统一y轴）--------
        fig, axes = plt.subplots(nrows=k, ncols=1, figsize=(fig_w, fig_h), sharex=True)
        if k == 1:
            axes = [axes]
        for i, node_idx in enumerate(bottom_idx):
            ax = axes[i]
            ax.plot(t, ts[:, node_idx], linewidth=2.0, color=cmap(i % cmap.N))
            ax.set_ylim(y_lo, y_hi)
            ax.set_title(f"Bottom-{i+1}: Node {node_idx}")
            ax.grid(alpha=0.3, linestyle='--')
            if i == k - 1:
                ax.set_xlabel("Time Index")
            ax.set_ylabel("Speed")
        plt.tight_layout()
        bottom_grid_path = os.path.join(
            self.importance_vis_dir,
            f"{self.dataset_name}_timeseries_bottom_grid_{tag}_sample{sample_idx}.png"
        )
        fig.savefig(bottom_grid_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        self._logger.info(f"[Bottom Grid RAW] Saved: {bottom_grid_path}")


    def save_importance_distribution(self, importance, epoch, step, tag: str = "overall"):
        """
        保存重要性分布直方图（文件名包含 tag）。
        >>> 调整：把图例移到图外右侧，避免与统计文字/图形重叠；统计信息放到轴内左上角。
        """
        os.makedirs(self.importance_vis_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(importance, bins=50, color='#1f77b4', alpha=0.7, label='Importance Distribution')
        ax.axvline(x=np.mean(importance), color='r', linestyle='--', label=f'Mean: {np.mean(importance):.4f}')
        ax.set_title(f'{self.dataset_name.upper()} Importance Distribution [{tag}]\nEpoch {epoch}, Step {step}',
                    fontsize=12, pad=20)
        ax.set_xlabel('Importance Score'); ax.set_ylabel('Frequency')

        # 统计信息：放在轴内左上角，避免与图例重叠
        stats_text = (f'Max: {np.max(importance):.4f}\n'
                    f'Min: {np.min(importance):.4f}\n'
                    f'Std: {np.std(importance):.4f}')
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

        # 图例放到图外右侧
        leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
        # 为右侧图例留白，同时保证导出时包含图例
        plt.tight_layout()
        filename = f"{self.dataset_name}_importance_dist_{tag}_epoch{epoch}_step{step:05d}.png"
        fig.savefig(os.path.join(self.importance_vis_dir, filename), dpi=400, bbox_inches='tight')
        plt.close(fig)

    def save_topk_timeseries(self, importance_scores, raw_x, sample_idx, tag="overall"):
        """保存Top-K节点时间序列图，文件名中包含tag。"""
        # importance_scores: [num_nodes]
        k = min(self.topk, len(importance_scores))
        topk_idx = np.argsort(importance_scores)[-k:][::-1]

        # 转换为原始值
        ts_data = raw_x[0, :, :, 0]  # shape: [T, N]
        ts_data = self.standard_scaler.inverse_transform(ts_data)

        plt.figure(figsize=(12, 6))
        cmap = plt.get_cmap('tab10')  # 可换成 'tab20', 'Set2', 'Paired' 等
        for i, node_idx in enumerate(topk_idx):
            plt.plot(ts_data[:, node_idx],
                    label=f'Node {node_idx}',
                    color=cmap(i % cmap.N))


        plt.title(f"Top-{k} Nodes Time Series [{tag}] Sample {sample_idx}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        out_name = f"{self.dataset_name}_topk_timeseries_{tag}_sample{sample_idx}.png"
        out_path = os.path.join(self.importance_vis_dir, out_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=400)
        plt.close()
        self._logger.info(f"[TopK Timeseries] Saved: {out_path}")


    def save_topk_timeseries_from_raw(self,
                                  importance_scores: np.ndarray,
                                  raw_speed: np.ndarray,
                                  sample_idx: int,
                                  tag: str = "overall",
                                  last_len: int = 288*14):
        """
        使用原始数据 raw_speed 绘制 Top-K 节点的时间序列（不使用训练/评估 batch）。
        - importance_scores: [N]
        - raw_speed: (T, N) 原始速度数据
        - last_len: 若为 None 或非法值，则绘制全长度
        - 颜色使用自动配色（tab10）
        """
        assert raw_speed.ndim == 2, "raw_speed must be (T, N)"
        T, N = raw_speed.shape
        k = min(self.topk, len(importance_scores))
        topk_idx = np.argsort(importance_scores)[-k:][::-1]

        # 全长度或最后 last_len
        if last_len is None or last_len <= 0 or last_len > T:
            s, e = 0, T
        else:
            s, e = max(0, T - last_len), T

        ts = raw_speed[s:e, :]  # (L, N)
        L = ts.shape[0]
        t = np.arange(L)

        import matplotlib.pyplot as plt
        plt.rcParams['xtick.labelsize'] = 16  # x轴刻度字体大小
        plt.rcParams['ytick.labelsize'] = 16  # y轴刻度字体大小
        plt.figure(figsize=(12, 6))

        cmap = plt.get_cmap('tab10')
        for i, node_idx in enumerate(topk_idx):
            plt.plot(t, ts[:, node_idx],
                    label=f'Node {node_idx}',
                    linewidth=2.0,
                    color=cmap(i % cmap.N))

        #plt.title(f"Top-{k} Nodes Time Series from RAW [{tag}] Sample {sample_idx}")
        plt.xlabel("Time Index")
        plt.ylabel("Value")
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(loc='best')

        out_name = f"{self.dataset_name}_topk_timeseries_{tag}_sample{sample_idx}.png"
        out_path = os.path.join(self.importance_vis_dir, out_name)
        os.makedirs(self.importance_vis_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=400, bbox_inches='tight')
        plt.close()
        self._logger.info(f"[TopK Timeseries RAW] Saved: {out_path}")

    def evaluate_node_importance(self, dataset='test', num_samples=1, raw_data_path: str = None):
        """
        评估并可视化节点重要性。
        现在的 Top/Bottom-5 时间序列图来自 raw_data_path 指向的原始数据，保存全长度。
        """
        self._logger.info(f"Evaluating node importance for {dataset} dataset")

        # ===== 加载原始数据 =====
        if raw_data_path is None:
            if self.dataset_name.startswith("metr"):
                raw_data_path = "data/METR-LA/metr-la.h5"
            elif self.dataset_name.startswith("pems-bay"):
                raw_data_path = "data/PEMS-BAY/pems-bay.h5"
            elif self.dataset_name.startswith("pems"):
                raw_data_path = "data/PEMS04/pems04.npz"
        try:
            raw_speed = self._load_raw_speed(raw_data_path)  # (T, N)
            if raw_speed.shape[1] != self.num_nodes:
                self._logger.warning(
                    f"[RAW] N mismatch: raw N={raw_speed.shape[1]} vs model N={self.num_nodes}. "
                    f"{'Trunc to model N' if raw_speed.shape[1] > self.num_nodes else 'Pad to model N'}."
                )
                if raw_speed.shape[1] > self.num_nodes:
                    raw_speed = raw_speed[:, :self.num_nodes]
                else:
                    pad = np.tile(raw_speed[:, [-1]], (1, self.num_nodes - raw_speed.shape[1]))
                    raw_speed = np.hstack([raw_speed, pad])
            self._logger.info(f"[RAW] Loaded raw data for TS plot: {raw_data_path}, shape={raw_speed.shape}")
        except Exception as e:
            self._logger.error(f"[RAW] Failed to load raw data: {e}")
            raw_speed = None

        try:
            data_loader = self._data[f'{dataset}_loader'].get_iterator()
            sample_count = 0
            for raw_x, raw_y in data_loader:
                if sample_count >= num_samples:
                    break
                if raw_x.shape[0] == 0:
                    continue

                x, y = self._prepare_data(raw_x, raw_y)

                # 触发一次前向，缓存中间变量
                if isinstance(self.amodel, iGraphformer):
                    _ = self.amodel(x)
                else:
                    _ = self.amodel(x, self.graph)

                # 取第一层编码器
                if not (hasattr(self.amodel, 'iG_encoder') and hasattr(self.amodel.iG_encoder, 'attn_layers')):
                    self._logger.warning("No iG_encoder.attn_layers found. Skip importance.")
                    return
                layer = self.amodel.iG_encoder.attn_layers[3]

                

                # last_W_q_2
                if getattr(layer, 'last_W_q_2', None) is not None:
                    s2_wq_nodes = self._node_scores_from_Wq(layer.last_W_q_2)
                    self.visualize_node_importance(s2_wq_nodes, 0, sample_count, tag="last_W_q_2")
                    self.save_importance_distribution(s2_wq_nodes, 0, sample_count, tag="last_W_q_2")
                    if raw_speed is not None:
                        self.save_extreme_timeseries_from_raw(
                            s2_wq_nodes, raw_speed, sample_count, tag="last_W_q_2", k=5, last_len=288*7
                        )

                

                sample_count += 1

            # 合并报告（可选）
            self.generate_importance_report()
        except Exception as e:
            self._logger.error(f"Error in node importance evaluation: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_importance_report(self):
        """
        将可视化图片合并为 PDF 报告。
        - 时间序列部分：仅收录“Top 栅格汇总页”和“Bottom 栅格汇总页”，
        从而实现 Top 一页、Bottom 一页。
        - 单节点扁长型 PNG 仍会保存到硬盘，但默认不放入 PDF。
        """
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            pdf_path = os.path.join(self.importance_vis_dir, f"{self.dataset_name}_importance_report.pdf")
            with PdfPages(pdf_path) as pdf:
                # 1) 地理散点 & 分布（按需可保留）
                geo_and_dist_patts = [
                    f"{self.dataset_name}_importance_*_epoch*.png",
                    f"{self.dataset_name}_importance_dist_*_epoch*.png",
                ]
                pages = []
                for p in geo_and_dist_patts:
                    pages += sorted(glob.glob(os.path.join(self.importance_vis_dir, p)))
                for f in pages:
                    img = plt.imread(f)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img); plt.axis('off')
                    plt.title(os.path.basename(f), fontsize=8)
                    pdf.savefig(); plt.close()

                # 2) 只加入“Top/Bottom 栅格汇总页”，各一页
                grid_patts = [
                    f"{self.dataset_name}_timeseries_top_grid_*_sample*.png",
                    f"{self.dataset_name}_timeseries_bottom_grid_*_sample*.png",
                ]
                grid_pages = []
                for p in grid_patts:
                    grid_pages += sorted(glob.glob(os.path.join(self.importance_vis_dir, p)))
                # 保持顺序：Top 在前，Bottom 在后
                for f in grid_pages:
                    img = plt.imread(f)
                    # 用图像原比例铺满一页
                    h, w = img.shape[:2]
                    # 设定页面尺寸与图像长宽比更匹配
                    base_w = 12
                    base_h = max(6, int(base_w * (h / max(w, 1))))
                    plt.figure(figsize=(base_w, base_h))
                    plt.imshow(img); plt.axis('off')
                    plt.title(os.path.basename(f), fontsize=8)
                    pdf.savefig(); plt.close()

            self._logger.info(f"[Report] Generated: {pdf_path}")
            print(f"[Report] Generated: {pdf_path}")
        except Exception as e:
            self._logger.error(f"Error generating importance report: {str(e)}")

    # ---------------- 数据准备 ----------------
    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        x0 = x[..., 0]
        y0 = y[..., 0]
        x = torch.from_numpy(x0).float()
        y = torch.from_numpy(y0).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        return x, y
    
    def _load_raw_speed(self, path: str) -> np.ndarray:
        """
        读取 METR-LA (.h5/.npz) 或 PEMS04/PEMS08 (.npz) 的速度数据。
        返回 shape=(T, N) 的 ndarray。
        """
        if path.endswith('.npz'):
            with np.load(path, allow_pickle=True) as data:
                if 'x' in data:      # METR-LA 预处理格式: (T, N, F)
                    x = data['x']
                elif 'data' in data: # PEMS04/08 格式: (T, N, F)
                    x = data['data']
                else:
                    raise KeyError("Unknown npz structure: no 'x' or 'data' key found")
            speed = x[..., 0]
        elif path.endswith('.h5'):
            import pandas as pd
            df = pd.read_hdf(path)   # 列为节点
            speed = df.values
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return speed  # (T, N)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='iGraphformer')
    #parser.add_argument('--config_filename', default='data/log/iGraphformer/20250813_131716_iGraphformer_l_12_h_12_lr_0.001_bs_64/_data/METR-LA/config_ig_metr.yaml')
    parser.add_argument('--config_filename', default='data/log/iGraphformer/20250813_164044_iGraphformer_l_12_h_12_lr_0.001_bs_64/_data/PEMS-BAY/config_ig_pems.yaml')
    parser.add_argument('--use_cpu_only', default=False, type=bool)
    parser.add_argument('--load_pretrained', default=True, type=bool, help='Whether to load a pretrained model.')
    #parser.add_argument('--pretrained_model_dir', default='data/log/iGraphformer/20250813_131716_iGraphformer_l_12_h_12_lr_0.001_bs_64/_data/METR-LAmodels/epo13.tar', type=str, help='Directory of the pretrained model.')
    parser.add_argument('--pretrained_model_dir', default='data/log/iGraphformer/20250813_164044_iGraphformer_l_12_h_12_lr_0.001_bs_64/_data/PEMS-BAYmodels/epo90.tar', type=str, help='Directory of the pretrained model.')
    parser.add_argument('--cuda', default='cuda:0', type=str)
    args = parser.parse_args()

    with open(args.config_filename, encoding='utf-8') as f:
        supervisor_config = yaml.load(f, Loader=yaml.FullLoader)
        print('[Config] file:', args.config_filename)

        supervisor = ModelsSupervisor(args.models,
                                      args.pretrained_model_dir,
                                      args.config_filename,
                                      args.cuda,
                                      **supervisor_config)

        truths, predictions = supervisor.evaluate(dataset='test')

        # 保存预测输出（可选）
        predictions_subset = predictions
        truths_subset = truths
        columns = [f'truth_{i}' for i in range(truths_subset.shape[1])] + \
                  [f'prediction_{i}' for i in range(predictions_subset.shape[1])]
        non_zero_mask = ~(np.all(truths_subset == 0, axis=1))
        truths_subset = truths_subset[non_zero_mask]
        predictions_subset = predictions_subset[non_zero_mask]
        combined_data = np.hstack([truths_subset, predictions_subset])
        combined_df = pd.DataFrame(combined_data, columns=columns)
        output_file = os.path.join(supervisor._log_dir, 'model_output.csv')
        combined_df.to_csv(output_file, index=False, header=True)
        print("[Export] Model outputs saved to:", output_file)

        # 生成 PDF 报告
        supervisor.generate_importance_report()


if __name__ == "__main__":
    main()


