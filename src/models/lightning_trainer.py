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

import fnmatch
import numpy as np
import pytorch_lightning as pt
import torch
import torch.optim as optim
import torch.nn.functional as F
import utilities.metric as M
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from torch.utils.data import DataLoader

from datasets.multidomain_dataset import Multidomain_Dataset
from datasets.utils import get_dataset
from utilities import metric as M
from utilities.file_utils import Utils as utils


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

        # Loss function for classification (not contrastive)
        self.loss_fn = torch.nn.CrossEntropyLoss()

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

        for (input_ids, attention_mask), label in batch:
            # Only convert to tensor if input is not already a tensor
            input_ids_tensor = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.long)
            attention_mask_tensor = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.tensor(attention_mask, dtype=torch.long)
            
            if not isinstance(label, torch.Tensor):
                label_tensor = torch.tensor(label, dtype=torch.long)
            else:
                label_tensor = label
            
            input_ids_list.append(input_ids_tensor)
            attention_mask_list.append(attention_mask_tensor)
            labels_list.append(label_tensor.view(-1))  # Flatten labels if needed

        input_ids = torch.stack(input_ids_list)         # [B, N, E, L]
        attention_mask = torch.stack(attention_mask_list)
        labels = torch.stack(labels_list)               # [B, N * E]

        return (input_ids, attention_mask), labels


    def validation_collate_fn(self, batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        #print(f">>> [validation_collate_fn] Received batch of size {len(batch)}", flush=True)
        for (input_ids, attention_mask), label in batch:
            input_ids_tensor = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.long)
            attention_mask_tensor = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.tensor(attention_mask, dtype=torch.long)
            
            if not isinstance(label, torch.Tensor):
                label_tensor = torch.tensor(label, dtype=torch.long)
            else:
                label_tensor = label

            input_ids_list.append(input_ids_tensor)
            attention_mask_list.append(attention_mask_tensor)
            labels_list.append(label_tensor.view(-1))  # Flatten labels if needed

        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        labels = torch.stack(labels_list)

        return (input_ids, attention_mask), labels

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
        batch_size = 1 if self.params.dataset_name in ["raw_amazon", "pan_paragraph", "pan21"] else self.params.batch_size

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
                pin_memory=self.params.pin_memory
            ),
            DataLoader(
                targets,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.params.num_workers,
                pin_memory=self.params.pin_memory
            )
        ]

    ##################################################
    # Training Methods
    ##################################################
    
    def _internal_step(self, batch, split_name):
        try:
            _, comment_embeddings = self._model_forward(batch)  # comment_embeddings: [B, N, E, D]
            transitions = comment_embeddings[:, :, 1:, :] - comment_embeddings[:, :, :-1, :]  # [B, N, E-1, D]
            transitions = transitions.reshape(-1, transitions.size(-1))  # [B * N * (E-1), D]

            _, labels = batch  # shape: [B, N, E-1]
            labels = labels.reshape(-1).long()  # [B * N * (E-1)]

            logits = self.classifier(transitions)
            loss = self.loss_fn(logits, labels)

            # Binary predictions: logits > 0 -> class 1 else 0
            preds = torch.argmax(logits, dim=1)
    
            # Compute accuracy or recall at threshold 0.5
            correct = (preds == labels).sum()
            accuracy = correct.float() / labels.size(0)
    
            num_classes = logits.shape[1]
            topk = min(8, num_classes)
            topk_preds = torch.topk(logits, k=topk, dim=1).indices
            matches = (topk_preds == labels.unsqueeze(1)).any(dim=1).float()  # [B], 1 if label in top8 else 0
            recall_at_8 = matches.mean()  # scalar
    
            embeddings_key = f"{split_name}embedding"  # e.g. "test_embedding", "val_embedding", "train_embedding"
    
            return {
                f"{split_name}loss": loss,
                f"{split_name}accuracy": accuracy,
                f"{split_name}R@8": recall_at_8,
                "logits": logits,
                "preds": preds,
                "ground_truth": labels,
                embeddings_key: transitions  # use dynamic key here
            }
    
        except Exception as e:
            print(f"Exception in _internal_step: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise e
    

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
        out = self._internal_step(batch, split_name="test_")
        self.log("test_R@8", out["test_R@8"])
        self.log("test_loss", out["test_loss"])
        return out

    def training_step_end(self, step_outputs):
        """Calculate the loss function according to the embeddings from every step."""
        episode_embeddings = step_outputs["train_embedding"].float()
        labels = step_outputs["ground_truth"]

        if labels.ndim > 1:
            labels = torch.argmax(labels, dim=1)

        logits = self.classifier(episode_embeddings)
        loss = self.loss_fn(logits, labels)

        return_dict = {"loss": loss}

        self.log("loss", return_dict["loss"], 
                on_step=True, on_epoch=True, prog_bar=True,
                batch_size=episode_embeddings.size(0))
        
        return return_dict

    def validation_epoch_end(self, outputs):
        """Calculates accuracy and average loss from validation outputs."""

        #print(">>> outputs type:", type(outputs))
        #print(">>> outputs len:", len(outputs))
        #print(">>> outputs[0] keys:", outputs[0].keys())

        if len(outputs) == 0:
            return

        # Concatenate predictions and labels
        all_preds = torch.cat([x['preds'] for x in outputs], dim=0)
        all_gt = torch.cat([x['ground_truth'] for x in outputs], dim=0)

        # Compute accuracy
        acc = (all_preds == all_gt).float().mean()

        # Average validation loss
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Log them
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_loss", val_loss, prog_bar=True)

    def test_epoch_end(self, outputs):
        """Calculates ranking + classification metrics from test step outputs."""
        logs = {}
    
        # === Flatten if nested (e.g. outputs = [[dict, dict], [dict, dict]]) ===
        if isinstance(outputs[0], list):
            outputs = [item for sublist in outputs for item in sublist]
    
        test_embeddings = []
        ground_truths = []
        logits_list = []
    
        for batch in outputs:
            test_embeddings.append(batch["test_embedding"])     # [B, D]
            ground_truths.append(batch["ground_truth"])         # [B]
            logits_list.append(batch["logits"])                 # [B, C]
    
        # Construct examples for ranking metrics
        examples = []
        for emb, gt in zip(test_embeddings, ground_truths):
            for e, g in zip(emb, gt):
                examples.append({
                    "test_embedding": e,
                    "ground_truth": g
                })
    
        examples = examples[:10000]  # Optional: limit to avoid OOM
        print(f">>> Total examples passed to compute_metrics: {len(examples)}")
    
        # === Compute Ranking Metrics ===
        ranking_metrics = M.compute_metrics(examples, examples, 'test')
    
        # === Compute Classification Metrics ===
        all_logits = torch.cat(logits_list, dim=0).cpu()        # [N, C]
        all_labels = torch.cat(ground_truths, dim=0).cpu()      # [N]
    
        probs = F.softmax(all_logits.float(), dim=1)
        preds = torch.argmax(probs, dim=1)
    
        acc = accuracy_score(all_labels, preds)
        prec, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='macro')
    
        try:
            roc_auc = roc_auc_score(all_labels, probs.numpy(), multi_class='ovr')
        except ValueError:
            roc_auc = float('nan')  # Handle edge case: only one class in labels
    
        classification_metrics = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc
        }
    
        # === Combine and log all metrics ===
        all_metrics = {**ranking_metrics, **classification_metrics}
        for k, v in all_metrics.items():
            self.log(f'test_{k}', v, batch_size=self.params.batch_size)
    
        scores = utils.dict2string(all_metrics, f'{self.params.experiment_id} version {self.logger.version}')
        mode = 'a' if os.path.exists(self.experiment_log_filename) else 'w'
        with open(self.experiment_log_filename, mode, encoding="utf-8") as f:
            f.write(scores)
    
        print(">>> Final test metrics:")
        print(scores)
    
    
    
