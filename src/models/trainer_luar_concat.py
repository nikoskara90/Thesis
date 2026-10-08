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

        self.experiment_log_filename = os.path.join(
            utils.output_path, 
            'experiments.log'
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
    def _internal_step(self, batch, split_name, mode="train"):
        # Forward pass: model may return a precomputed loss
        # Expecting: episode_embeddings, comment_embeddings, logits, pair_labels, loss, problem_ids
        model_out = self._model_forward(batch)

        # Unpack carefully to support older model signatures (backwards-compatible)
        # Common return: episode_embeddings, comment_embeddings, logits, pair_labels, loss, problem_ids
        if len(model_out) == 6:
            _, comment_embeddings, logits_all, labels_list, model_loss, problem_ids = model_out
        elif len(model_out) == 5:
            # older signature without loss
            _, comment_embeddings, logits_all, labels_list, problem_ids = model_out
            model_loss = None
        else:
            # Defensive: attempt best-effort unpack
            try:
                _, comment_embeddings, logits_all, labels_list, model_loss, problem_ids = model_out
            except Exception as e:
                raise RuntimeError(f"_model_forward returned unexpected number of items: {len(model_out)}") from e

        B, N, E, D = comment_embeddings.shape
        comment_embeddings = F.normalize(comment_embeddings, dim=-1, eps=1e-8)

        pair_emb_list, labels_pairs_list, problem_map_list = [], [], []

        for b in range(B):
            emb_seq = comment_embeddings[b]  # [N, E, D]
            # labels_list here may be either:
            # - direct flattened pair labels per batch element: [P_pairs] (as before), or
            # - the original labels object (per-author shape) depending on model version
            lbl_seq = labels_list[b] if labels_list is not None else None
            pid = int(problem_ids[b]) if problem_ids is not None else b

            flat_emb = emb_seq.reshape(-1, D)  # [N*E, D]

            # If lbl_seq is not flattened, try to flatten similarly to previous logic
            # We expect lbl_seq to be a 1D tensor length P_pairs (N*E - 1) with -100 fill for invalid pairs
            if lbl_seq is None:
                # no labels for this batch element => skip
                continue

            # Ensure it's a tensor on same device
            lbl_seq = lbl_seq.to(flat_emb.device)

            # Mask of valid paragraph pairs
            valid_mask = (lbl_seq != -100)
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

            if pid == 1:
                print(f"[DEBUG] === Problem 1 ===")
                print(f"comment_embeddings[b].shape: {emb_seq.shape}")
                print(f"labels_list[b].shape: {lbl_seq.shape}")
                print(f"valid_indices: {valid_indices}")

            if valid_indices.numel() == 0:
                continue  # no valid pairs in this problem

            # Create pairs along the flattened sequence
            emb1 = flat_emb[valid_indices]           # current paragraph
            emb2 = flat_emb[valid_indices + 1]       # next paragraph
            labels_pairs = lbl_seq[valid_indices]    # corresponding labels

            # Concatenate for classifier
            pair_emb = torch.cat([emb1, emb2], dim=-1)
            pair_emb_list.append(pair_emb)
            labels_pairs_list.append(labels_pairs)
            problem_map_list.append(torch.full((labels_pairs.size(0),), pid,
                                            device=flat_emb.device, dtype=torch.long))

        # If no valid pairs across batch, return a neutral output
        if len(pair_emb_list) == 0:
            return None

        # Concatenate all pairs from all problems in batch
        pair_emb = torch.cat(pair_emb_list, dim=0)          # [total_pairs, 2*D]
        labels_pairs = torch.cat(labels_pairs_list, dim=0)  # [total_pairs]
        problem_map = torch.cat(problem_map_list, dim=0)    # [total_pairs]

        # If model returned logits for every pair, prefer them. Otherwise compute via trainer.classifier (if present)
        if logits_all is not None and logits_all.numel() != 0:
            # logits_all: [B, P-1, C] → select only valid pairs, flatten
            # Build indexing mask to pick entries corresponding to valid pairs across batch
            # We will collect logits for valid indices similarly to how we collected pair_emb_list
            selected_logits = []
            start = 0
            for b in range(B):
                # number of pairs in this problem = N*E - 1
                P_pairs = (N * E) - 1
                logits_b = logits_all[b]  # [P_pairs, C]
                # find which indices we used earlier for this b
                # Reconstruct valid indices from labels_list[b]
                lbl_seq = labels_list[b].to(logits_b.device)
                valid_mask_b = (lbl_seq != -100)
                valid_idx_b = torch.nonzero(valid_mask_b, as_tuple=False).squeeze(-1)
                if valid_idx_b.numel() == 0:
                    continue
                selected_logits.append(logits_b[valid_idx_b])
            if len(selected_logits) == 0:
                return None
            logits = torch.cat(selected_logits, dim=0)  # [total_pairs, C]
        else:
            # Fall back: use trainer-level classifier (only if it's defined)
            if hasattr(self, "classifier") and self.classifier is not None:
                logits = self.classifier(pair_emb)                  # [total_pairs, 2]
            else:
                raise RuntimeError("No logits returned by model and trainer.classifier not available.")

        # If model produced a loss, use it; otherwise compute CE here
        if model_loss is not None:
            loss = model_loss
        else:
            loss = F.cross_entropy(logits, labels_pairs)

        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == labels_pairs).float().mean()
        val_R8 = accuracy if split_name.startswith("val") else torch.tensor(0.0, device=comment_embeddings.device)

        # Debug prints for Problem 1
        if (problem_map == 1).any():
            mask = (problem_map == 1)
            print(f"\n[DEBUG] Problem predictions")
            print(f"  logits: {logits[mask].detach().cpu().numpy()}")
            print(f"  preds: {preds[mask].detach().cpu().numpy()}")
            print(f"  ground_truth: {labels_pairs[mask].detach().cpu().numpy()}")

        return {
            f"{split_name}loss": loss,
            f"{split_name}accuracy": accuracy,
            f"{split_name}R@8": val_R8,
            "logits": logits,
            "preds": preds,
            "ground_truth": labels_pairs,
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
        # The classification loss is already computed in _internal_step as 'train_loss'
        loss = step_outputs["train_loss"]

        # Log loss for this step and epoch
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=step_outputs["ground_truth"].size(0)  # ensures proper averaging
        )

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        """Validation summary for concatenated embedding + classification head."""
        # Filter out None outputs (batches with no valid pairs)
        outputs = [x for x in outputs if x is not None]
        if len(outputs) == 0:
            return

        # Concatenate logits and ground-truth
        all_logits = torch.cat([x["logits"] for x in outputs], dim=0)          # [num_pairs_total, 2]
        all_gt = torch.cat([x["ground_truth"] for x in outputs], dim=0)       # [num_pairs_total]

        # Predictions
        all_preds = torch.argmax(all_logits, dim=-1)

        # Compute metrics
        accuracy = (all_preds == all_gt).float().mean()
        self.log("val_accuracy", accuracy, prog_bar=True)

        try:
            val_loss = torch.stack([x[f"val_loss"] for x in outputs]).mean()
        except KeyError:
            val_loss = torch.tensor(0.0, device=all_gt.device)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("validation_R@8", accuracy, prog_bar=True)

        # scikit-learn metrics (safe zero_division)
        gt_np = all_gt.cpu().numpy()
        preds_np = all_preds.cpu().numpy()
        prec = precision_score(gt_np, preds_np, average='macro', zero_division=0)
        rec = recall_score(gt_np, preds_np, average='macro', zero_division=0)
        f1 = f1_score(gt_np, preds_np, average='macro', zero_division=0)

        self.log("val_precision", prec)
        self.log("val_recall", rec)
        self.log("val_f1", f1)

        # Optional: debug print for first batch or summary
        print(f"[VAL] Accuracy={accuracy:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")


    def test_epoch_end(self, outputs):
        """Saves predictions to JSON files using classifier head outputs."""
        save_dir = os.path.join("output_json", f"{self.experiment_id}", f"{self.logger.version or 0}")
        os.makedirs(save_dir, exist_ok=True)

        # Flatten outputs from multiple devices/batches
        flat_outputs = [b for o in outputs for b in (o if isinstance(o, list) else [o])]

        for batch in flat_outputs:
            logits = batch.get("logits", None)
            problem_ids = batch.get("problem_ids", None)

            # Safety guards
            if logits is None or logits.numel() == 0:
                print(f"[DEBUG] Skipping batch with empty logits for problem_ids={problem_ids}")
                continue
            if problem_ids is None or len(problem_ids) == 0:
                print(f"[DEBUG] Skipping batch with no problem_ids (logits_len={logits.numel()})")
                continue

            # Determine predicted class
            preds = torch.argmax(logits, dim=-1).cpu().numpy()   # 0=same, 1=change

            # Optional debug summary
            num_changes = int(preds.sum())
            #print(f"[DEBUG] Problem_ids={problem_ids} logits_len={logits.shape[0]} -> predicted_changes={num_changes}")

            # Save to JSON (one file per problem)
            pid = str(problem_ids[0])
            safe_pid = re.sub(r'[\\/*?:"<>|]', '_', pid)
            filename = f"solution-problem-{safe_pid}.json"
            filepath = os.path.join(save_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"changes": preds.tolist()}, f)

            #print(f"[DEBUG] Saved predictions for Problem {pid} -> {filename}")

        print(f">>> All solution files saved to: {save_dir}")

