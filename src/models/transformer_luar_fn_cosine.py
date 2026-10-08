import os
import torch
import sys
import torch.nn as nn
from functools import partial
from einops import rearrange, reduce, repeat
from models.layers import MemoryEfficientAttention, SelfAttention
from models.trainer_luar_cosine import LightningTrainer as LightningTrainerLUAR
from utilities.file_utils import Utils as utils
import torch.nn.functional as F
from transformers import AutoModel, RobertaTokenizer, AutoTokenizer  # Ensure tokenizer is available

class Transformer(LightningTrainerLUAR):
    """Defines a Transformer model for author style representation."""
    
    def __init__(self, params, tokenizer=None):
        super(Transformer, self).__init__(params)
        self.save_hyperparameters()

        self.create_transformer()
        self.experiment_id = params.experiment_id if hasattr(params, "experiment_id") else "default_exp"
        self.learning_rate = params.learning_rate
        self.attn_fn = SelfAttention()
        self.margin = getattr(params, "contrastive_margin", 0.5)
        self.alpha = getattr(params, "contrastive_weight", 0.5)  # weight between CE & contrastive
        self.cosine_loss_fn = nn.CosineEmbeddingLoss(margin=self.margin)

    def create_transformer(self):
        """Creates the Transformer model."""
        
        transformer_modelnames = {
            "roberta": "sentence-transformers/paraphrase-distilroberta-base-v1",  # HF hub repo name
            "roberta_base": "roberta-base",
        }

        modelname = transformer_modelnames[self.params.model_type]
        self.params.pretrained_model_name_or_path = modelname

        model_folder = modelname.replace("sentence-transformers/", "")
        local_model_path = os.path.join(utils.transformer_path, model_folder)

        # Check if local folder exists
        if os.path.exists(local_model_path):
            print(f"[INFO] Loading model and tokenizer from local path: {local_model_path}")
            self.encoder = AutoModel.from_pretrained(local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.params.pretrained_model_name_or_path = local_model_path
        else:
            print(f"[INFO] Loading model and tokenizer from Hugging Face hub: {modelname}")
            self.encoder = AutoModel.from_pretrained(modelname, use_auth_token=False)
            self.tokenizer = AutoTokenizer.from_pretrained(modelname, use_auth_token=False)

        self.transformer = self.encoder
        # Set model config
        self.hidden_size = self.encoder.config.hidden_size
        self.num_attention_heads = self.encoder.config.num_attention_heads
        self.dim_head = self.hidden_size // self.num_attention_heads

        # === Force freeze encoder ===
        freeze_mode = getattr(self.params, "freeze_mode", "backbone")  
        if freeze_mode == "all":
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[INFO] All encoder parameters frozen. Model is feature extractor only.")
        elif freeze_mode == "backbone":
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[INFO] Encoder frozen, head will be trained.")
        else:
            print("[INFO] Training full encoder + head.")
                    
        # Optional: Replace attention function if necessary
        if self.params.attention_fn_name != "default":
            self.replace_attention()

        # Enable gradient checkpointing if set in params
        if self.params.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        token_embeddings: [B, L, hidden_size]
        attention_mask: [B, L]
        """
        # Expand mask to match embedding dimension
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=token_embeddings.size(-1)).float()
        
        # Detect rows that are fully padded (sum over seq_len)
        valid_mask = attention_mask.sum(dim=1) > 0  # [B], True for rows with at least 1 token

        # Initialize pooled tensor
        pooled = torch.zeros(token_embeddings.size(0), token_embeddings.size(-1), device=token_embeddings.device)

        if valid_mask.any():
            # Only consider valid rows
            valid_embeddings = token_embeddings[valid_mask]
            valid_mask_expanded = input_mask_expanded[valid_mask]

            sum_mask = valid_mask_expanded.sum(dim=1)  # [num_valid, hidden_size]
            pooled[valid_mask] = (valid_embeddings * valid_mask_expanded).sum(dim=1) / sum_mask.clamp(min=1.0)

        # Clamp extreme values
        return pooled.clamp(-1e4, 1e4)
            
    def compute_contrastive_loss(self, embeddings, labels):
        """
        embeddings: [B, N, E, D] or [1, N, E, D]
        labels: [B, N, E-1] or [N, E-1] (-100 for padding)
        """
        # Ensure embeddings always have batch dimension
        if embeddings.dim() == 3:  # [N, E, D]
            embeddings = embeddings.unsqueeze(0)  # [1, N, E, D]

        # Ensure labels always have batch dimension
        if labels is not None and labels.dim() == 2:  # [N, E-1]
            labels = labels.unsqueeze(0)  # [1, N, E-1]

        B, N, E, D = embeddings.shape
        emb_a_list, emb_b_list, targets_list = [], [], []

        for b in range(B):
            for n in range(N):
                for e in range(E - 1):
                    if e >= labels.shape[2]:  # safe now
                        continue
                    lbl = labels[b, n, e].item()
                    if lbl == -100:
                        continue

                    emb_a_list.append(embeddings[b, n, e])
                    emb_b_list.append(embeddings[b, n, e + 1])
                    targets_list.append(1.0 if lbl == 0 else -1.0)

                # Cross-author
                if n < N - 1:
                    if labels[b, n, -1].item() != -100 and labels[b, n + 1, 0].item() != -100:
                        emb_a_list.append(embeddings[b, n, -1])
                        emb_b_list.append(embeddings[b, n + 1, 0])
                        lbl_cross = labels[b, n, -1].item()
                        targets_list.append(1.0 if lbl_cross == 0 else -1.0)

        if len(emb_a_list) == 0:
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            targets = torch.tensor([], device=embeddings.device)
            return loss, targets

        emb_a_tensor = F.normalize(torch.stack(emb_a_list, dim=0), p=2, dim=-1)
        emb_b_tensor = F.normalize(torch.stack(emb_b_list, dim=0), p=2, dim=-1)
        targets = torch.tensor(targets_list, device=embeddings.device, dtype=torch.float32)

        loss = self.cosine_loss_fn(emb_a_tensor, emb_b_tensor, targets)
        return loss, targets

    def get_episode_embeddings(self, data, chunk_size=128):
        input_ids, attention_mask = data  # [N, E, L] or [B, N, E, L]

        # Ensure batch dimension
        if input_ids.dim() == 3:  # [N, E, L] → batch size 1
            input_ids = input_ids.unsqueeze(0)         # [1, N, E, L]
            attention_mask = attention_mask.unsqueeze(0)

        B, N, E, L = input_ids.shape
        assert input_ids.shape == attention_mask.shape, f"Mismatch {input_ids.shape} vs {attention_mask.shape}"

        # Flatten all authors & paragraphs
        flattened_input_ids = input_ids.view(B * N * E, L)
        flattened_attention_mask = attention_mask.view(B * N * E, L)

        # Clamp token IDs & attention masks
        vocab_size = self.transformer.config.vocab_size
        flattened_input_ids = flattened_input_ids.clamp(0, vocab_size - 1)
        flattened_attention_mask = flattened_attention_mask.clamp(0, 1)

        all_comment_embeddings = []

        for start_idx in range(0, flattened_input_ids.size(0), chunk_size):
            end_idx = min(start_idx + chunk_size, flattened_input_ids.size(0))
            chunk_input_ids = flattened_input_ids[start_idx:end_idx]
            chunk_attention_mask = flattened_attention_mask[start_idx:end_idx]

            # debug shapes
            #print(f"[DEBUG] chunk {start_idx}:{end_idx} input_ids shape: {chunk_input_ids.shape}, "
             #   f"attention_mask shape: {chunk_attention_mask.shape}")

            if chunk_input_ids.size(0) == 0:
                print(f"[WARNING] Skipping empty chunk {start_idx}:{end_idx}")
                continue

            # Replace fully padded rows with CLS token
            pad_token_id = self.tokenizer.pad_token_id
            mask_empty_rows = (chunk_input_ids == pad_token_id).all(dim=1)
            if mask_empty_rows.any():
                cls_token_id = self.tokenizer.cls_token_id
               # print(f"[DEBUG] {mask_empty_rows.sum().item()} empty rows replaced with CLS token")
                chunk_input_ids[mask_empty_rows] = cls_token_id
                chunk_attention_mask[mask_empty_rows] = 1

            # Transformer forward
            outputs = self.transformer(chunk_input_ids, attention_mask=chunk_attention_mask)
            embeddings = outputs.last_hidden_state  # [chunk_size, L, hidden_size]

            # Check NaNs/Infs
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                print(f"[WARNING] NaN/Inf in transformer outputs! chunk {start_idx}:{end_idx}")
                embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e3, neginf=-1e3)

            pooled_embeddings = self.mean_pooling(embeddings, chunk_attention_mask)

            #print(f"[DEBUG] pooled_embeddings shape: {pooled_embeddings.shape}, "
             #   f"mean={pooled_embeddings.mean().item():.6f}, std={pooled_embeddings.std().item():.6f}")

            # Append pooled embeddings
            all_comment_embeddings.append(pooled_embeddings)
            torch.cuda.empty_cache()

        if len(all_comment_embeddings) == 0:
            #print("[ERROR] all_comment_embeddings is empty! Likely an empty batch or no valid paragraphs.")
            #print(f"flattened_input_ids shape: {flattened_input_ids.shape}")
            #print(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
            # create a dummy tensor to avoid crashing
            comment_embeddings = torch.zeros(B, N, E, self.hidden_size, device=input_ids.device)
        else:
            # Concatenate and reshape back to [B, N, E, hidden_size]
            comment_embeddings = torch.cat(all_comment_embeddings, dim=0)
            comment_embeddings = comment_embeddings.view(B, N, E, -1)

        # Episode-level embedding: [B, N*E*hidden_size]
        episode_embeddings = comment_embeddings.view(B, N * E * comment_embeddings.size(-1))

        return episode_embeddings, comment_embeddings

    def get_dataloader(self, split, dataset_class):
        """Dynamically instantiate a dataset class and return the DataLoader."""
        dataset = dataset_class(self.params, split, self.tokenizer)
        
        return torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=(split == "train"))
    
    def forward(self, data, classify=False):
        """Defines the forward pass to obtain episode and comment embeddings."""
        episode_embeddings, comment_embeddings = self.get_episode_embeddings(data)
        if classify:
            logits = self.classifier(episode_embeddings)
            return logits
        else:
            return episode_embeddings, comment_embeddings

    def _model_forward(self, batch, mode="train"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", None)
        problem_ids = batch.get("problem_ids", None)

        # Forward pass
        episode_embeddings, comment_embeddings = self.forward((input_ids, attention_mask))

        if  labels is not None:
            # Compute contrastive loss for training
            cosine_loss, pair_labels = self.compute_contrastive_loss(comment_embeddings, labels)
        else:
            # Testing: skip contrastive loss, return dummy
            cosine_loss = torch.tensor(0.0, device=comment_embeddings.device, requires_grad=False)
            pair_labels = torch.tensor([], device=comment_embeddings.device)

        return episode_embeddings, comment_embeddings, pair_labels, cosine_loss, problem_ids

    def replace_attention(self):
        """Replaces the Transformer's Attention mechanism.
        
           NOTE: This feature has only been tested with the regular 
                 original LUAR SBERT pretrained-model.
        """ 
        attn_fn = {
            "memory_efficient": partial(
                MemoryEfficientAttention, 
                q_bucket_size=16, k_bucket_size=32, 
                heads=self.num_attention_heads, dim_head=self.dim_head,
            ),
        }
        
        for i, layer in enumerate(self.transformer.encoder.layer):
            attention = attn_fn[self.params.attention_fn_name]()
            
            state_dict = layer.attention.self.state_dict()
            attention.load_state_dict(state_dict, strict=True)

            self.transformer.encoder.layer[i].attention.self = attention

