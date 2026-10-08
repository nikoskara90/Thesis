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

        self.best_cosine_threshold = 0.5  # initial default
        self.best_val_f1 = 0.0

        self.loss_fn = torch.nn.CrossEntropyLoss()

        embedding_dim = self.params.hidden_dim  # D
        num_classes = getattr(self.params, "num_classes", 2)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, num_classes)
        )

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
        # Get embeddings from model [B, 2, D]
        embeddings, labels, problem_ids = self._model_forward(batch, mode=mode)

        # Flatten labels if needed
        if labels.ndim > 1:
            labels = labels.view(-1)
        labels = labels.long()

        # Concatenate the two embeddings
        combined = torch.cat([embeddings[:, 0, :], embeddings[:, 1, :]], dim=-1)  # [B, 2D]

        # Forward through classifier
        logits = self.classifier(combined)  # [B, num_classes]

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        # Predictions
        preds = torch.argmax(logits, dim=1)

        # Build output dictionary
        out = {
            f"{split_name}loss": loss,
            "logits": logits,
            "preds": preds,
            "ground_truth": labels,
            f"{split_name}embedding": combined,
        }

        if mode == "test":
            out["problem_ids"] = problem_ids

        return out


    def training_step(self, batch, batch_idx):
        out = self._internal_step(batch, split_name="train_")
        batch_size = out["ground_truth"].size(0)  
        self.log("train_loss", out["train_loss"], batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True)
        return out


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._internal_step(batch, split_name="val_")
        batch_size = out["ground_truth"].size(0)
        self.log("val_loss", out["val_loss"], batch_size=batch_size, on_epoch=True)
        return out


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._internal_step(batch, split_name="test_", mode="test")
        return out


    def training_step_end(self, step_outputs):
        return {"loss": step_outputs["train_loss"]}


    def validation_epoch_end(self, outputs):
        """Logs validation loss and dummy R@8 for checkpoint compatibility."""
        if len(outputs) == 0:
            return

        # Average validation loss
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Dummy R@8 (required for checkpoint callback)
        validation_R8 = torch.tensor(0.0, device=val_loss.device)

        # Log them
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("validation_R@8", validation_R8, prog_bar=True)



    def test_epoch_end(self, outputs):
        import re  # ensure re is imported at the top if not already

        save_dir = os.path.join("output_json", f"{self.experiment_id}", f"{self.logger.version or 0}")
        os.makedirs(save_dir, exist_ok=True)

        # Flatten outputs in case there are multiple dataloaders or nested lists
        flat_outputs = [b for o in outputs for b in (o if isinstance(o, list) else [o])]

        for batch in flat_outputs:
            problem_ids = batch["problem_ids"]
            preds = batch["preds"].cpu().numpy().tolist()  

            # Save one file per problem
            pid = str(problem_ids[0])
            safe_pid = re.sub(r'[\\/*?:"<>|]', '_', pid)
            filename = f"solution-problem-{safe_pid}.json"
            filepath = os.path.join(save_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"changes": preds}, f)

        print(f">>> Solution files saved to: {save_dir}")



