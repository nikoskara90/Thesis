# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
from abc import ABC, abstractmethod
from itertools import chain
from math import ceil
import logging
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

logger = logging.getLogger("test_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    log_path = os.path.join("logs", "test_luar_metrics.log")
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
        self.margin = getattr(params, "margin", 0.5)
        self.best_cosine_threshold = 0.5  # initial default
        self.best_val_f1 = 0.0
        self.best_cosine_mode = "ge"

        self.experiment_log_filename = os.path.join(
            utils.output_path, 
            'experiments.log'
        )

        # Loss function for classification (not contrastive)
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
        # Only 1 episode per batch, so batch size = 1
        item = batch[0]
        input_ids = item["input_ids"]         # [N, E_author, L]
        attention_mask = item["attention_mask"]
        labels = item["labels"]               # list of tensors per author
        problem_ids = item["problem_ids"]
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,                # keep as list
            "problem_ids": problem_ids,
        }
    
    def validation_collate_fn(self, batch):
        # Only 1 episode per batch
        item = batch[0]
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        labels = item["labels"]              # list of tensors per author
        problem_ids = item["problem_ids"]
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,               # list of tensors per author
            "problem_ids": problem_ids,
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

        #print(">>> [val_dataloader] Dataset length:", len(val_data), flush=True)

        data_loader = DataLoader(
            val_data,
            batch_size=1,
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
        batch_size = 1 if self.params.dataset_name in ["raw_amazon", "pan_paragraph", "pan21"] else self.params.batch_size

        if "+" in self.params.dataset_name:
            queries = Multidomain_Dataset(self.params, "test", is_queries=True)
            targets = Multidomain_Dataset(self.params, "test", is_queries=False)
        else:
            queries, targets = get_dataset(self.params, split="test")

        return [
            DataLoader(
                queries,
                batch_size=1,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=self.params.pin_memory
            ),
            DataLoader(
                targets,
                batch_size=1,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=self.params.pin_memory
            )
        ]

    ##################################################
    # Training Methods
    ##################################################
    # Assume _model_forward returns:
    # comment_embeddings: [B, N, E, D] (for logging if needed)
    # labels_pairs_f: [num_pairs_total] (already flattened)
    # loss: computed cosine/contrastive loss
    def _internal_step(self, batch, split_name, mode="train"):

        # Forward pass (already returns loss + labels + embeddings)
        _, comment_embeddings, labels_pairs_f, loss, problem_ids = self._model_forward(batch, mode=mode)
        B, N, E, D = comment_embeddings.shape
        #print(f"[DEBUG] batch_idx info: B={B}, N={N}, E={E}, D={D}")

        # Embedding sanity check
        #debug_stats(comment_embeddings, "comment_embeddings (before normalization)")

        # Normalize
        comment_embeddings = F.normalize(comment_embeddings, dim=-1, eps=1e-8)
        #debug_stats(comment_embeddings, "comment_embeddings (after normalization)")

        if labels_pairs_f is not None and labels_pairs_f.numel() > 0:
            # Compute cosine similarities only for valid pairs
            emb1 = comment_embeddings[:, :, :-1, :].reshape(-1, D)
            emb2 = comment_embeddings[:, :, 1:, :].reshape(-1, D)

            cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)

            # Align with labels_pairs_f length
            cos_sim = cos_sim[:labels_pairs_f.numel()] 

            threshold = 0.5 if mode == "train" else getattr(self, "best_cosine_threshold", 0.5)
            preds = (cos_sim >= threshold).long()
            accuracy = (preds == labels_pairs_f.long()).float().mean()
        else:
            cos_sim = torch.tensor([], device=comment_embeddings.device)
            preds = torch.tensor([], device=comment_embeddings.device)
            labels_pairs_f = torch.tensor([], device=comment_embeddings.device)
            accuracy = torch.tensor(0.0, device=comment_embeddings.device)

        # Top-8
        if cos_sim.numel() > 0:
            k = min(8, cos_sim.numel())
            sorted_scores, indices = torch.topk(cos_sim, k=k)
            top_labels = labels_pairs_f[indices]
            val_R8 = top_labels.sum().float() / top_labels.numel()
        else:
            val_R8 = torch.tensor(0.0, device=comment_embeddings.device)

        #debug_stats(loss, f"{split_name}loss")

        return {
            f"{split_name}loss": loss,
            f"{split_name}accuracy": accuracy,
            f"{split_name}R@8": val_R8,
            "cosine_sim": cos_sim,
            "preds": preds,
            "ground_truth": labels_pairs_f,
            f"{split_name}embedding": comment_embeddings,
            "problem_ids": problem_ids
        }


    def training_step(self, batch, batch_idx):
        out = self._internal_step(batch, split_name="train_")
        self.log("train_loss", out["train_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_R@8", out["train_R@8"], on_epoch=True)
        return out
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._internal_step(batch, split_name="val_")
        self.log("validation_R@8", out["val_R@8"], on_epoch=True, prog_bar=True)
        self.log("validation_accuracy", out["val_accuracy"], on_epoch=True, prog_bar=True)
        self.log("val_loss", out["val_loss"], on_epoch=True)
        return out

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._internal_step(batch, split_name="test_", mode="test")
        self.log("test_R@8", out["test_R@8"])
        self.log("test_loss", out["test_loss"])
        return out

    def training_step_end(self, step_outputs):
        """
        For cosine similarity training:
        Uses the BCE loss computed in _internal_step directly.
        """
        # The BCE loss is already computed in _internal_step as 'train_loss'
        loss = step_outputs["train_loss"]

        # Log it
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=step_outputs["ground_truth"].size(0)
        )

        return {"loss": loss}
    
    def validation_epoch_end(self, outputs):
        """Validation summary with adaptive threshold search, metrics, and debug prints."""

        # Default metrics
        acc = val_loss = prec = rec = f1 = 0.0
        self.best_cosine_threshold = getattr(self, "best_cosine_threshold", 0.5)
        self.best_cosine_mode = getattr(self, "best_cosine_mode", "ge")
        self.best_val_f1 = getattr(self, "best_val_f1", 0.0)

        if len(outputs) == 0:
            print("[VAL] Warning: no validation outputs found.")
            self.log_metrics(acc, val_loss, prec, rec, f1)
            return

        # Gather all cosine similarities and ground-truth labels
        cos_list = [x['cosine_sim'] for x in outputs if 'cosine_sim' in x and x['cosine_sim'].numel() > 0]
        gt_list = [x['ground_truth'] for x in outputs if 'ground_truth' in x and x['ground_truth'].numel() > 0]

        if not cos_list or not gt_list:
            print("[VAL] Warning: no valid validation pairs found.")
            self.log_metrics(acc, val_loss, prec, rec, f1)
            return

        all_cos = torch.cat(cos_list)
        all_gt = torch.cat(gt_list)

        # Ensure matching lengths
        if all_cos.size(0) != all_gt.size(0):
            m = min(all_cos.size(0), all_gt.size(0))
            all_cos = all_cos[:m]
            all_gt = all_gt[:m]

        # Map ground-truth from {-1, 1} → {0, 1} to match predictions
        all_gt_mapped = all_gt.clone()
        all_gt_mapped[all_gt_mapped == -1] = 0
        all_gt_mapped[all_gt_mapped == 1] = 1

        # Threshold search (adaptive to cosine similarity range)
        same_author_cos = all_cos[all_gt_mapped == 0]  # 0 = same author
        change_author_cos = all_cos[all_gt_mapped == 1]  # 1 = different author


        if same_author_cos.numel() > 0 and change_author_cos.numel() > 0:
            mode_hint = "ge" if same_author_cos.mean() > change_author_cos.mean() else "lt"

            cos_min, cos_max = all_cos.min().item(), all_cos.max().item()
            thresholds = torch.linspace(cos_min, cos_max, steps=21)  # fewer steps for debug
            best_f1, best_t, best_mode = -1.0, self.best_cosine_threshold, mode_hint

            #print(f"[DEBUG-threshold] Searching thresholds from {cos_min:.4f} to {cos_max:.4f}, mode hint: {mode_hint}")

            for t in thresholds:
                for mode in ("ge", "lt"):
                    preds = (all_cos >= t).long() if mode == "ge" else (all_cos < t).long()
                    try:
                        f1_tmp = f1_score(all_gt_mapped.cpu().numpy(), preds.cpu().numpy(),
                                        average='macro', zero_division=0)
                        #print(f"[DEBUG-threshold] t={t:.4f}, mode={mode}, F1={f1_tmp:.4f}")
                        if f1_tmp > best_f1:
                            best_f1, best_t, best_mode = f1_tmp, t.item(), mode
                            print(f"  -> New best F1={best_f1:.4f} at t={best_t:.4f}, mode={best_mode}")
                    except Exception as e:
                        print(f"[DEBUG-threshold] Exception at t={t:.4f}, mode={mode}: {e}")


            self.best_cosine_threshold = best_t
            self.best_cosine_mode = best_mode
            self.best_val_f1 = best_f1
        else:
            print("[VAL] Warning: only one class present. Skipping threshold search.")

        # Final predictions
        all_preds = 1- (all_cos >= self.best_cosine_threshold).long() if self.best_cosine_mode == "ge" else (all_cos < self.best_cosine_threshold).long()
        # Accuracy
        acc = (all_preds == all_gt_mapped).float().mean().item()

        # Validation loss
        try:
            val_loss = torch.stack([x['val_loss'] for x in outputs if 'val_loss' in x]).mean().item()
        except:
            val_loss = 0.0

        # Precision, Recall, F1
        if all_gt_mapped.numel() > 0:
            prec = precision_score(all_gt_mapped.cpu().numpy(), all_preds.cpu().numpy(), average='macro', zero_division=0)
            rec = recall_score(all_gt_mapped.cpu().numpy(), all_preds.cpu().numpy(), average='macro', zero_division=0)
            f1 = f1_score(all_gt_mapped.cpu().numpy(), all_preds.cpu().numpy(), average='macro', zero_division=0)

        # Log metrics
        self.log_metrics(acc, val_loss, prec, rec, f1)
        print(f"[INFO] Best threshold this epoch: {self.best_cosine_threshold:.4f}, Mode: {self.best_cosine_mode}, F1: {self.best_val_f1:.4f}")


    def test_epoch_end(self, outputs):
        """Saves predictions to JSON files using best cosine threshold & mode, with debug prints."""
        save_dir = os.path.join("output_json", f"{self.experiment_id}", f"{self.logger.version or 0}")
        os.makedirs(save_dir, exist_ok=True)

        # Flatten outputs from multiple devices/batches
        flat_outputs = [b for o in outputs for b in (o if isinstance(o, list) else [o])]

        threshold = float(getattr(self, "best_cosine_threshold", 0.5))
        mode = getattr(self, "best_cosine_mode", "ge")

        print(f"[INFO] Testing with threshold={threshold:.3f}, mode={mode}")

        for batch_idx, batch in enumerate(flat_outputs):
            cos_sim = batch.get("cosine_sim", None)
            problem_ids = batch.get("problem_ids", None)

            if cos_sim is None or cos_sim.numel() == 0:
                print(f"[DEBUG-test] Batch {batch_idx} has empty cos_sim. Shape: {None if cos_sim is None else cos_sim.shape}")
                continue

            if problem_ids is None or len(problem_ids) == 0:
                print(f"[DEBUG-test] Batch {batch_idx} has empty problem_ids. Value: {problem_ids}")
                continue

            # flatten cosine_sim to 1D
            cos_sim = cos_sim.detach().cpu().view(-1)

            # Debug stats for this batch
            print(f"[DEBUG-test] Batch {batch_idx}: cos_sim_len={cos_sim.numel()} "
                f"min={float(cos_sim.min()):.4f} max={float(cos_sim.max()):.4f} "
                f"mean={float(cos_sim.mean()):.4f} std={float(cos_sim.std()):.4f}")


            # Compute predictions
            if mode == "ge":
                pred_same = (cos_sim >= threshold)
            else:
                pred_same = (cos_sim < threshold)
            pred_change = (~pred_same).long().cpu().numpy().tolist()

            # Print first few predictions for debug
            print(f"[DEBUG-test] Batch {batch_idx} first 10 predicted changes: {pred_change[:10]}")

            # Ensure problem_ids aligns: one JSON file per problem
            for i, pid in enumerate(problem_ids):
                safe_pid = re.sub(r'[\\/*?:"<>|]', '_', str(pid))
                filename = f"solution-problem-{safe_pid}.json"
                filepath = os.path.join(save_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump({"changes": pred_change}, f)

                # Debug print only for first problem
                if i == 0:
                    print(f"[DEBUG-test-first] Problem {pid}: predicted_changes={sum(pred_change)}")

        print(f">>> All solution files saved to: {save_dir}")


    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_cosine_threshold"] = getattr(self, "best_cosine_threshold", 0.5)
        checkpoint["best_cosine_mode"] = getattr(self, "best_cosine_mode", "ge")
        checkpoint["best_val_f1"] = getattr(self, "best_val_f1", 0.0)
        print(f"[INFO] Checkpoint save → threshold={self.best_cosine_threshold}, mode={self.best_cosine_mode}, F1={self.best_val_f1:.3f}")

    def on_load_checkpoint(self, checkpoint):
        self.best_cosine_threshold = checkpoint.get("best_cosine_threshold", 0.5)
        self.best_cosine_mode = checkpoint.get("best_cosine_mode", "ge")
        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        print(f"[INFO] Checkpoint load → threshold={self.best_cosine_threshold}, mode={self.best_cosine_mode}, F1={self.best_val_f1:.3f}")

    def log_metrics(self, acc, val_loss, prec, rec, f1):
        """Helper to log metrics cleanly."""
        self.log("val_accuracy", acc, prog_bar=True, on_epoch=True)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        self.log("validation_R@8", acc, prog_bar=True, on_epoch=True)
        self.log("val_best_threshold", float(self.best_cosine_threshold), prog_bar=True, on_epoch=True)
        self.log("val_best_mode", 0.0 if self.best_cosine_mode == "ge" else 1.0, on_epoch=True)
        self.log("val_best_f1", float(self.best_val_f1), prog_bar=True, on_epoch=True)
        self.log("val_precision", float(prec), on_epoch=True)
        self.log("val_recall", float(rec), on_epoch=True)
        self.log("val_f1_after", float(f1), prog_bar=True, on_epoch=True)