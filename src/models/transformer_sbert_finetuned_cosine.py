import os
import torch
import sys
import torch.nn as nn
from functools import partial
from einops import rearrange, reduce, repeat
from models.layers import MemoryEfficientAttention, SelfAttention
from models.trainer_sbert_cosine import LightningTrainer as LightningTrainerSBERT
from utilities.file_utils import Utils as utils
from transformers import AutoModel, RobertaTokenizer, AutoTokenizer  # Ensure tokenizer is available
import torch.nn.functional as F

class Transformer(LightningTrainerSBERT):
    """Defines a Transformer model for author style representation."""
    
    def __init__(self, params, tokenizer=None):
        super(Transformer, self).__init__(params)
        self.save_hyperparameters()
        self.create_transformer()
        self.learning_rate = params.learning_rate
        self.experiment_id = params.experiment_id if hasattr(params, "experiment_id") else "default_exp"
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
        """Applies mean pooling as described in SBERT."""
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=self.hidden_size).float()
        sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
        sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_pair_embeddings(self, data):
        input_ids, attention_mask = data  # [B, 2, L]
        B, P, L = input_ids.shape

        input_ids = input_ids.view(-1, L)
        attention_mask = attention_mask.view(-1, L)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)  # [B*2, D]
        embeddings = embeddings.view(B, P, -1)  # [B, 2, D]

        return embeddings

    def get_dataloader(self, split, dataset_class):
        """Dynamically instantiate a dataset class and return the DataLoader."""
        dataset = dataset_class(self.params, split, self.tokenizer)
        
        return torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=(split == "train"))
    
    def forward(self, data):
        # Always return raw embeddings for cosine similarity
        return self.get_pair_embeddings(data)  # shape: [B, 2, D]

    def _model_forward(self, batch, mode="train"):
        if isinstance(batch, dict):  # test path
            (input_ids, attention_mask) = batch["inputs"]
            labels = batch["labels"]
            problem_ids = batch.get("problem_id", None)
        else:  # train/val path
            (input_ids, attention_mask), labels, problem_ids = batch

        # directly pass input_ids [B, 2, L] to forward
        embeddings = self.forward((input_ids, attention_mask))  # should return [B, 2, D]

        return embeddings, labels, problem_ids

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

import os
import torch
import sys
import torch.nn as nn
from functools import partial
from einops import rearrange, reduce, repeat
from models.layers import MemoryEfficientAttention, SelfAttention
from models.trainer_sbert_cosine import LightningTrainer as LightningTrainerSBERT
from utilities.file_utils import Utils as utils
from transformers import AutoModel, RobertaTokenizer, AutoTokenizer  # Ensure tokenizer is available
import torch.nn.functional as F

class Transformer(LightningTrainerSBERT):
    """Defines a Transformer model for author style representation."""
    
    def __init__(self, params, tokenizer=None):
        super(Transformer, self).__init__(params)
        self.save_hyperparameters()
        self.create_transformer()
        self.learning_rate = params.learning_rate
        self.experiment_id = params.experiment_id if hasattr(params, "experiment_id") else "default_exp"
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
        """Applies mean pooling as described in SBERT."""
        input_mask_expanded = repeat(attention_mask, 'b l -> b l d', d=self.hidden_size).float()
        sum_embeddings = reduce(token_embeddings * input_mask_expanded, 'b l d -> b d', 'sum')
        sum_mask = torch.clamp(reduce(input_mask_expanded, 'b l d -> b d', 'sum'), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_pair_embeddings(self, data):
        input_ids, attention_mask = data  # [B, 2, L]
        B, P, L = input_ids.shape

        input_ids = input_ids.view(-1, L)
        attention_mask = attention_mask.view(-1, L)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)  # [B*2, D]
        embeddings = embeddings.view(B, P, -1)  # [B, 2, D]

        return embeddings

    def get_dataloader(self, split, dataset_class):
        """Dynamically instantiate a dataset class and return the DataLoader."""
        dataset = dataset_class(self.params, split, self.tokenizer)
        
        return torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size, shuffle=(split == "train"))
    
    def forward(self, data):
        # Always return raw embeddings for cosine similarity
        return self.get_pair_embeddings(data)  # shape: [B, 2, D]

    def _model_forward(self, batch, mode="train"):
        if isinstance(batch, dict):  # test path
            (input_ids, attention_mask) = batch["inputs"]
            labels = batch["labels"]
            problem_ids = batch.get("problem_id", None)
        else:  # train/val path
            (input_ids, attention_mask), labels, problem_ids = batch

        # directly pass input_ids [B, 2, L] to forward
        embeddings = self.forward((input_ids, attention_mask))  # should return [B, 2, D]

        return embeddings, labels, problem_ids

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

