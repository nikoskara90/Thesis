import os
import torch
import sys
import torch.nn as nn
from functools import partial
from einops import rearrange, reduce, repeat
from models.layers import MemoryEfficientAttention, SelfAttention
from models.trainer_luar_concat import LightningTrainer as LightningTrainerLUAR
from utilities.file_utils import Utils as utils
from transformers import AutoModel, RobertaTokenizer, AutoTokenizer  # Ensure tokenizer is available

class Transformer(LightningTrainerLUAR):
    """Defines a Transformer model for author style representation (concat + classification head)."""
    
    def __init__(self, params, tokenizer=None):
        super(Transformer, self).__init__(params)
        self.save_hyperparameters()

        self.create_transformer()
        # classifier: small MLP (recommended over single Linear)
        self.hidden_dim = self.hidden_size
        head_hidden = getattr(self.params, "head_hidden_dim", self.hidden_dim)
        head_dropout = getattr(self.params, "head_dropout", 0.1)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, self.params.num_labels)
        )
        # cross-entropy with ignore_index handled in trainer when computing loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.experiment_id = params.experiment_id if hasattr(params, "experiment_id") else "default_exp"
        self.learning_rate = params.learning_rate
        self.attn_fn = SelfAttention()

    def create_transformer(self):
        """Creates the Transformer model."""
        
        transformer_modelnames = {
            "roberta": "sentence-transformers/paraphrase-distilroberta-base-v1",
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
        """Safe mean pooling that avoids NaN/Inf."""
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=self.hidden_size).float()
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)  # [b, d]
        pooled = (token_embeddings * input_mask_expanded).sum(1) / sum_mask
        pooled = pooled.clamp(-1e3, 1e3)
        return pooled

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

        # ==== SAFETY: clamp token IDs & attention masks ====
        vocab_size = self.transformer.config.vocab_size
        flattened_input_ids = flattened_input_ids.clamp(0, vocab_size - 1)
        flattened_attention_mask = flattened_attention_mask.clamp(0, 1)

        all_comment_embeddings = []

        for start_idx in range(0, flattened_input_ids.size(0), chunk_size):
            end_idx = min(start_idx + chunk_size, flattened_input_ids.size(0))
            chunk_input_ids = flattened_input_ids[start_idx:end_idx]
            chunk_attention_mask = flattened_attention_mask[start_idx:end_idx]

            # Skip completely padded rows (all PAD)
            pad_token_id = self.tokenizer.pad_token_id
            mask_empty_rows = (chunk_input_ids == pad_token_id).all(dim=1)
            if mask_empty_rows.any():
                cls_token_id = self.tokenizer.cls_token_id
                chunk_input_ids[mask_empty_rows] = cls_token_id
                chunk_attention_mask[mask_empty_rows] = 1

            outputs = self.transformer(chunk_input_ids, attention_mask=chunk_attention_mask)
            embeddings = outputs.last_hidden_state  # [chunk, L, hidden_size]

            # Mean pooling
            pooled_embeddings = self.mean_pooling(embeddings, chunk_attention_mask)
            all_comment_embeddings.append(pooled_embeddings)
            torch.cuda.empty_cache()

        # Concatenate and reshape back to [B, N, E, hidden_size]
        comment_embeddings = torch.cat(all_comment_embeddings, dim=0)
        comment_embeddings = comment_embeddings.view(B, N, E, -1)

        # Episode-level embedding: flatten if needed (legacy)
        episode_embeddings = comment_embeddings.view(B, N * E * comment_embeddings.size(-1))

        return episode_embeddings, comment_embeddings
    
    def classify_pairs_from_embeddings(self, emb_a, emb_b):
        """
        emb_a, emb_b: [B, P, D] or [B, D] or [P, D]
        Returns: logits shaped like input pairs: [B, P, C] or [B, C] or [P, C]
        """
        assert emb_a.shape == emb_b.shape, f"Mismatch emb shapes {emb_a.shape} vs {emb_b.shape}"
        orig_shape = emb_a.shape[:-1]   # e.g., (B, P) or (B,) or (P,)
        D = emb_a.size(-1)
        flat_a = emb_a.view(-1, D)
        flat_b = emb_b.view(-1, D)

        cat = torch.cat([flat_a, flat_b], dim=-1)  # [M, 2D]
        logits = self.classifier(cat)              # [M, C]

        return logits.view(*orig_shape, -1)

    def get_dataloader(self, split, dataset_class):
        dataset = dataset_class(self.params, split, self.tokenizer)
        return torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=(split == "train"))
    
    def forward(self, data):
        """Return episode and comment embeddings (no classification here)."""
        episode_embeddings, comment_embeddings = self.get_episode_embeddings(data)
        return episode_embeddings, comment_embeddings

    def _model_forward(self, batch):
        """
        Returns:
          episode_embeddings: [B, N*E*D] (legacy flattened)
          comment_embeddings: [B, N, E, D]
          logits: [B, P_pairs, C] where P_pairs == (N*E - 1)
          pair_labels: [B, P_pairs] with -100 where pair is invalid (cross-author boundary)
          problem_ids
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", None)            # dataset provides [B, N, E-1] (after collate)
        problem_ids = batch.get("problem_ids", None)

        episode_embeddings, comment_embeddings = self.forward((input_ids, attention_mask))
        B, N, E, D = comment_embeddings.shape
        flat = comment_embeddings.view(B, N * E, D)   # [B, P] where P = N*E

        # Build consecutive paragraph pairs: (p0->p1, p1->p2, ..., p_{P-2}->p_{P-1})
        emb_a = flat[:, :-1, :]   # [B, P-1, D]
        emb_b = flat[:, 1:, :]    # [B, P-1, D]

        # Build aligned pair labels from labels provided by dataset (labels per author: [B, N, E-1])
        # We'll place those into a flattened pair vector of length P-1 and set -100 at cross-author boundaries.
        device = flat.device
        P = N * E
        P_pairs = P - 1
        pair_labels = torch.full((B, P_pairs), -100, dtype=torch.long, device=device)

        if labels is not None:
            # ensure batch dim
            # expected labels shape from dataset collate: [B, N, E-1] OR [N, E-1] for single sample
            if labels.dim() == 2:
                labels_batch = labels.unsqueeze(0)  # [1, N, E-1]
            else:
                labels_batch = labels  # [B, N, E-1]

            # If labels_batch dims mismatched with N, try to adapt (defensive)
            if labels_batch.shape[1] != N:
                # try transpose or attempt best-effort reshape
                labels_batch = labels_batch.view(labels_batch.size(0), N, -1)

            # Fill pair_labels for each author block
            for a in range(N):
                start_par = a * E            # index of first paragraph of author a in flat
                start_pair = start_par      # pair starting at paragraph index start_par -> start_par+1
                # we can place E-1 labels into positions [start_pair, start_pair + E-1)
                label_slice = labels_batch[:, a, :].to(device)  # [B, E-1]
                pair_labels[:, start_pair:start_pair + (E - 1)] = label_slice

        # Classify pairs
        logits = self.classify_pairs_from_embeddings(emb_a, emb_b)  # [B, P-1, C]

        return episode_embeddings, comment_embeddings, logits, pair_labels, problem_ids

    def replace_attention(self):
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
