# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import sys
import json
from abc import ABC, abstractmethod
from itertools import chain
from math import ceil
import logging
import os

from collections import defaultdict
import re
import fnmatch
import numpy as np
import pytorch_lightning as pt
import torch
import torch.optim as optim
import torch.nn.functional as F
import utilities.metric as M
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from torch.utils.data import DataLoader

from datasets.multidomain_dataset import Multidomain_Dataset
from datasets.utils import get_dataset
from utilities import metric as M
from utilities.file_utils import Utils as utils

# Setup logger once (in __init__ or module level)
logger = logging.getLogger("test_logger")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if this function is called multiple times
if not logger.handlers:
    log_path = os.path.join("logs", "test_metrics.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class LightningTrainer(pt.LightningModule, ABC):
    """Defines all the PyTorch Lightning training functions. 
       Our model (SBERT), inherits from this class.
    """
    @abstractmethod
    def _model_forward(self, batch):
        """Passes a batch of data through the model. 
           This method must be implemented within the model.
        """
        pass
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.learning_rate = getattr(params, "learning_rate", 1e-4) 
        self.margin = getattr(params, "margin", 0.3)
    
        self.best_cosine_threshold = 0.5  # initial default
        self.best_val_f1 = 0.0
        self.best_cosine_mode = "ge"

        self.loss_fn = torch.nn.CosineEmbeddingLoss(margin=self.margin)

        # Optional: keep this if you still want to experiment with contrastive learning
        self.contrastive_loss = losses.SupConLoss(
            temperature=self.params.temperature, 
            distance=CosineSimilarity()
        )

    def configure_optimizers(self):
        """Configures the LR Optimizer & Scheduler.
        """
        if self.params.learning_rate_scaling:
            lr_factor = np.power(self.params.batch_size / 32, 0.5)
        else:
            lr_factor = 1
            
        learning_rate = self.learning_rate * lr_factor
        print("Using LR: {}".format(learning_rate))
        optimizer = optim.AdamW(
            chain(self.parameters()), lr=learning_rate)
            
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=300, eta_min=0.0001)

        return [optimizer], [scheduler]
    
    def train_collate_fn(self, batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        problem_ids_list = []

        for item in batch:
            input_ids, attention_mask = item["inputs"]
            labels = item["labels"]
            problem_id = item["problem_id"]

            input_ids_list.append(input_ids)       # [num_pairs, 2, L]
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)             # [num_pairs]
            problem_ids_list.append(problem_id)

        # Concatenate paragraph pairs across the batch
        input_ids = torch.cat(input_ids_list, dim=0)         # [total_pairs, 2, L]
        attention_mask = torch.cat(attention_mask_list, dim=0)
        labels = torch.cat(labels_list, dim=0)              # [total_pairs]

        return (input_ids, attention_mask), labels, problem_ids_list

    def validation_collate_fn(self, batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        problem_ids_list = []

        for item in batch:
            input_ids, attention_mask = item["inputs"]
            labels = item["labels"]
            problem_id = item["problem_id"]

            input_ids_list.append(input_ids)       # [num_pairs, 2, L]
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)             # [num_pairs]
            problem_ids_list.append(problem_id)

        # Concatenate paragraph pairs across the batch
        input_ids = torch.cat(input_ids_list, dim=0)         # [total_pairs, 2, L]
        attention_mask = torch.cat(attention_mask_list, dim=0)
        labels = torch.cat(labels_list, dim=0)              # [total_pairs]

        return (input_ids, attention_mask), labels, problem_ids_list

    def test_collate_fn(self, batch):
        item = batch[0]  # single problem
        input_ids, attention_mask = item["inputs"]
        labels = item["labels"]
        problem_id = item["problem_id"]

        return {
            "inputs": (input_ids, attention_mask),
            "labels": labels,
            "problem_id": [problem_id] * labels.size(0)  # repeat per pair
        }


    def train_dataloader(self):
        """Returns the training DataLoader."""
        if "+" in self.params.dataset_name:
            train_dataset = Multidomain_Dataset(self.params, "train")
        else:
            train_dataset = get_dataset(self.params, split="train")

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
            collate_fn=self.train_collate_fn  # Use the custom collate function here
        )

        return data_loader

    def val_dataloader(self):
        """Returns the validation DataLoader."""
        batch_size = 1 if self.params.dataset_name in ["raw_amazon", "pan_paragraph", "pan21"] else self.params.batch_size

        if "+" in self.params.dataset_name:
            val_data = Multidomain_Dataset(self.params, "validation")
        else:
            val_data = get_dataset(self.params, split="validation", only_queries=True)

        data_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.params.pin_memory,
            num_workers=self.params.num_workers,
            collate_fn=self.validation_collate_fn  # Ensure the custom collate function is used here too
        )

        return data_loader

    def load_multidomain_data(self, split):
        """Handles loading of multi-domain data."""
        data_path = os.path.join(self.params.dataset_path, f"{split}")
        problems = self.load_data_from_text_files(data_path, "problem-*.txt")
        truths = self.load_data_from_json_files(data_path, "truth-problem-*.json")
        
        combined_data = self.combine_problems_truths(problems, truths)
        return combined_data

    def load_data_from_text_files(self, directory, pattern):
        """Load all text files matching the given pattern from a specified directory."""
        files = [f for f in os.listdir(directory) if fnmatch.fnmatch(f, pattern)]
        data = []
        
        if not files:
            print(f"No files found matching the pattern: {pattern}")
            
        for file in files:
            try:
                with open(os.path.join(directory, file), 'r', encoding="utf-8") as f:
                    data.append(f.read())  # Store the content of each text file
            except IOError as e:
                print(f"Error opening {file}: {e}")
                
        return data


    def load_data_from_json_files(self, directory, pattern):
        """Load all JSON files matching the given pattern from a specified directory."""
        files = [f for f in os.listdir(directory) if fnmatch.fnmatch(f, pattern)]
        data = []
        
        if not files:
            print(f"No files found matching the pattern: {pattern}")
            
        for file in files:
            try:
                with open(os.path.join(directory, file), 'r', encoding="utf-8") as f:
                    data.append(json.load(f))  # Store the content of each JSON file
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error opening or decoding {file}: {e}")
        
        return data


    def combine_problems_truths(self, problems, truths):
        """Combine problem texts and truth JSONs into a dataset."""
        combined_data = []
        for problem, truth in zip(problems, truths):
            combined_data.append((problem, truth))
        return combined_data

    def test_dataloader(self):
        batch_size = 1

        if "+" in self.params.dataset_name:
            queries = Multidomain_Dataset(self.params, "test", is_queries=True)
            targets = Multidomain_Dataset(self.params, "test", is_queries=False)
        else:
            queries, targets = get_dataset(self.params, split="test")

        return [
            DataLoader(
                queries,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=self.params.pin_memory,
                collate_fn=self.test_collate_fn
            ),
            DataLoader(
                targets,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=self.params.pin_memory,
                collate_fn=self.test_collate_fn
            )
        ]

    ##################################################
    # Training Methods
    ##################################################
    def _internal_step(self, batch, split_name, mode="train"):
        embeddings, labels, _ = self._model_forward(batch, mode=mode)  # [B, E, D]
        B, E, D = embeddings.shape

        # Flatten consecutive paragraph pairs
        embedding_1 = embeddings[:, :-1, :].reshape(-1, D)
        embedding_2 = embeddings[:, 1:, :].reshape(-1, D)

        # Normalize embeddings
        embedding_1 = F.normalize(embedding_1, dim=-1, eps=1e-8)
        embedding_2 = F.normalize(embedding_2, dim=-1, eps=1e-8)

        # Map labels: 0 (same) -> +1, 1 (change) -> -1
        labels_mapped = torch.where(labels.view(-1) == 0, 1.0, -1.0)

        # --- Loss ---
        loss = self.loss_fn(embedding_1, embedding_2, labels_mapped)

        # --- Cosine similarity ---
        cos_sim = F.cosine_similarity(embedding_1, embedding_2, dim=-1)  # in [-1,1]

        if split_name.startswith("val") or split_name.startswith("test"):
            print(f"[INTERNAL] Batch size: {B}")
            print(f"[INTERNAL] Embedding 1 mean: {embedding_1.mean().item():.4f}, std: {embedding_1.std().item():.4f}")
            print(f"[INTERNAL] Embedding 2 mean: {embedding_2.mean().item():.4f}, std: {embedding_2.std().item():.4f}")
            print(f"[INTERNAL] Cosine sim: min={cos_sim.min().item():.4f}, max={cos_sim.max().item():.4f}, mean={cos_sim.mean().item():.4f}")
            print(f"[INTERNAL] Labels mapped: {labels_mapped[:2].cpu().numpy()}")

        # --- Prediction ---
        if mode == "train":
            threshold = 0.5
            pred_mode = "lt"
        else:
            threshold = getattr(self, "best_cosine_threshold", 0.5)
            pred_mode = getattr(self, "best_cosine_mode", "ge")

        if pred_mode == "ge":
            preds = (cos_sim >= threshold).long()   # predict same (0) if cos_sim ≥ threshold
        else:
            preds = (cos_sim < threshold).long()    # predict change (1) if cos_sim < threshold

        # Accuracy (compare with raw labels in {0,1})
        accuracy = (preds == labels.view(-1)).float().mean()

        out = {
            f"{split_name}loss": loss,
            f"{split_name}accuracy": accuracy,
            "cosine_sim": cos_sim,
            "preds": preds,
            "ground_truth": labels,
            f"{split_name}embedding": embeddings,
        }

        if mode == "test":
            out["problem_ids"] = batch["problem_id"]

        if split_name.startswith("val") or split_name.startswith("test"):
            print(f"[DEBUG] Predictions (first 20): {preds[:10].cpu().numpy()}")
            print(f"[DEBUG] Ground truth (first 20): {labels.view(-1)[:10].cpu().numpy()}")

        return out

    def training_step(self, batch, batch_idx):
        out = self._internal_step(batch, split_name="train_")
        batch_size = out["ground_truth"].size(0)  
        self.log("train_loss", out["train_loss"], batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", out["train_accuracy"], batch_size=batch_size, on_epoch=True)
        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._internal_step(batch, split_name="val_")
        batch_size = out["ground_truth"].size(0)
        self.log("val_accuracy", out["val_accuracy"], batch_size=batch_size, on_epoch=True, prog_bar=True)
        self.log("val_loss", out["val_loss"], batch_size=batch_size, on_epoch=True)
        return out

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._internal_step(batch, split_name="test_", mode="test")
        return out

    def training_step_end(self, step_outputs):
        """Calculate the loss function according to the embeddings from every step."""
        return {"loss": step_outputs["train_loss"]}

    def validation_epoch_end(self, outputs):
        """Calculates accuracy, average loss, and stores best cosine similarity threshold."""

        if len(outputs) == 0:
            return

        all_cos = torch.cat([x['cosine_sim'] for x in outputs])
        all_gt = torch.cat([x['ground_truth'] for x in outputs])

        # Map 0=same author, 1=change
        same_author_cos = all_cos[all_gt == 0]  # 0 = same
        change_author_cos = all_cos[all_gt == 1] # 1 = change

        pos_mean = same_author_cos.mean().item()
        neg_mean = change_author_cos.mean().item()

        print(f"[DEBUG] Same-author cosine mean={pos_mean:.4f}, std={same_author_cos.std():.4f}")
        print(f"[DEBUG] Change-author cosine mean={neg_mean:.4f}, std={change_author_cos.std():.4f}")

        # Determine correct prediction mode automatically
        mode_hint = "ge" if pos_mean > neg_mean else "lt"

        thresholds = torch.linspace(-1, 1, steps=201)
        best_f1, best_t, best_mode = 0.0, 0.5, mode_hint

        for t in thresholds:
            for mode in ["ge", "lt"]:
                preds = (all_cos >= t).long() if mode=="ge" else (all_cos < t).long()
                f1 = f1_score(all_gt.cpu().numpy(), preds.cpu().numpy(), zero_division=0)
                #print(f"[DEBUG] Threshold={t:.2f}, Mode={mode}, F1={f1:.4f}")
                if f1 > best_f1:
                    best_f1, best_t, best_mode = f1, t.item(), mode

        # Save best threshold & mode
        self.best_cosine_threshold = best_t
        self.best_cosine_mode = best_mode
        self.best_val_f1 = best_f1

        # --- Use best threshold & mode for reporting ---
        if self.best_cosine_mode == "ge":
            all_preds = (all_cos >= self.best_cosine_threshold).long()
        else:
            all_preds = (all_cos < self.best_cosine_threshold).long()

        acc = (all_preds == all_gt).float().mean()
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log("val_accuracy", acc, prog_bar=True)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("validation_R@8", acc, prog_bar=True)

        self.log("val_best_threshold", self.best_cosine_threshold)
        self.log("val_best_mode", 0.0 if self.best_cosine_mode=="ge" else 1.0)
        self.log("val_best_f1", self.best_val_f1)

        precision = precision_score(all_gt.cpu().numpy(), all_preds.cpu().numpy())
        recall = recall_score(all_gt.cpu().numpy(), all_preds.cpu().numpy())
        self.log("val_precision", precision)
        self.log("val_recall", recall)

        print(f"[INFO] Best threshold this epoch: {self.best_cosine_threshold:.3f}, "
            f"Mode: {self.best_cosine_mode}, F1: {self.best_val_f1:.3f}")


    def test_epoch_end(self, outputs):
        save_dir = os.path.join("output_json", f"{self.experiment_id}", f"{self.logger.version or 0}")
        os.makedirs(save_dir, exist_ok=True)

        flat_outputs = [b for o in outputs for b in (o if isinstance(o, list) else [o])]

        threshold = getattr(self, "best_cosine_threshold", 0.5)

        for batch in flat_outputs:
            cos_sim = batch["cosine_sim"]
            problem_ids = batch["problem_ids"]

            mode = getattr(self, "best_cosine_mode", "ge")
            if mode == "ge":
                preds = (cos_sim >= threshold).long().cpu().numpy().tolist()
            else:
                preds = (cos_sim < threshold).long().cpu().numpy().tolist()

            pid = str(problem_ids[0])
            safe_pid = re.sub(r'[\\/*?:"<>|]', '_', pid)
            filename = f"solution-problem-{safe_pid}.json"
            filepath = os.path.join(save_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"changes": preds}, f)

        print(f">>> Solution files saved to: {save_dir}")
        print(f"[INFO] Using best cosine threshold: {threshold:.3f}")

    # Called when saving a checkpoint
    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_cosine_threshold"] = getattr(self, "best_cosine_threshold", 0.5)
        print(f"[INFO] Best threshold saved={self.best_cosine_threshold}")

    # Called when loading a checkpoint
    def on_load_checkpoint(self, checkpoint):
        self.best_cosine_threshold = checkpoint.get("best_cosine_threshold", 0.5)
        print(f"[INFO] Restored threshold={self.best_cosine_threshold}")

