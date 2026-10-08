# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances

def compute_metrics(
    queries: torch.cuda.FloatTensor, 
    targets: torch.cuda.FloatTensor, 
    split: str
) -> dict:
    """Computes all the metrics specified through the cmd-line. 

    Args:
        queries (list of dict): Each dict must contain 'ground_truth' and '<split>_embedding'.
        targets (list of dict): Same structure as queries.
        split (str): One of "validation", "test", etc.
    """
    # Get query and target authors (robust to scalar or 1D tensor ground_truth)
    query_authors = torch.cat([
        x['ground_truth'].view(-1) for x in queries
    ]).cpu().numpy()

    target_authors = torch.cat([
        x['ground_truth'].view(-1) for x in targets
    ]).cpu().numpy()

    # Get all query and target embeddings
    q_list = torch.stack([
        x[f'{split}_embedding'] for x in queries
    ]).cpu().numpy()
    
    t_list = torch.stack([
        x[f'{split}_embedding'] for x in targets
    ]).cpu().numpy()
    
    # Compute metrics
    metric_scores = ranking(q_list, t_list, query_authors, target_authors)
    return metric_scores
    
def ranking(
        queries,
        targets,
        query_authors,
        target_authors,
        metric='cosine',
        batch_size=512,
    ):
        num_queries = len(query_authors)
        ranks = np.zeros(num_queries, dtype=np.float32)
        reciprocal_ranks = np.zeros(num_queries, dtype=np.float32)
    
        # Defensive reshape for targets
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
    
        for start in range(0, num_queries, batch_size):
            end = min(start + batch_size, num_queries)
            q_batch = queries[start:end]
    
            if q_batch.ndim == 1:
                q_batch = q_batch.reshape(1, -1)
    
            if metric == 'cosine':
                dists = cosine_distances(q_batch, targets)
            else:
                from sklearn.metrics import pairwise_distances
                dists = pairwise_distances(q_batch, targets, metric=metric)
    
            for i, dist in enumerate(dists):
                global_i = start + i
                sorted_idx = np.argsort(dist)
                sorted_authors = target_authors[sorted_idx]
                rank = np.where(sorted_authors == query_authors[global_i])[0][0]
                ranks[global_i] = rank
                reciprocal_ranks[global_i] = 1.0 / (rank + 1)
    
        return {
            'R@8': np.mean(ranks <= 8),
            'R@16': np.mean(ranks <= 16),
            'R@32': np.mean(ranks <= 32),
            'R@64': np.mean(ranks <= 64),
            'MRR': np.mean(reciprocal_ranks)
        }

