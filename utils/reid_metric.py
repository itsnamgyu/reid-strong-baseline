# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import json
import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        # NAMGYU
        print("Saving output features and dataset metadata")
        os.makedirs("outputs", exist_ok=True)
        np.save("outputs/qf.npy", qf.cpu().numpy())
        np.save("outputs/gf.npy", gf.cpu().numpy())
        with open("outputs/other.json", "w") as f:
            other = {
                "q_pids": q_pids.tolist(),
                "g_pids": g_pids.tolist(),
                "q_camids": q_camids.tolist(),
                "g_camids": g_camids.tolist(),
            }
            json.dump(other, f)

        gf = gf.cpu()
        qf = qf.cpu()

        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()

        # NAMGYU
        print("Saving distmat")
        np.save("outputs/distmat.npy", distmat)

        # NAMGYU
        print("Evaluating...")
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        print("Evaluation complete")


        return cmc, mAP


class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()

        # NAMGYU
        print("Saving output features and dataset metadata")
        os.makedirs("outputs", exist_ok=True)
        np.save("outputs/qf.npy", qf.cpu().numpy())
        np.save("outputs/gf.npy", gf.cpu().numpy())
        with open("outputs/other.json", "w") as f:
            other = {
                "q_pids": q_pids.tolist(),
                "g_pids": g_pids.tolist(),
                "q_camids": q_camids.tolist(),
                "g_camids": g_camids.tolist(),
            }
            json.dump(other, f)

        gf = gf.cpu()
        qf = qf.cpu()

        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        # NAMGYU
        print("Saving distmat")
        np.save("outputs/distmat.npy", distmat)

        # NAMGYU
        print("Evaluating...")
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        print("Evaluation complete")

        return cmc, mAP
