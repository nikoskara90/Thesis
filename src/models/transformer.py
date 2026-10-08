import os
import torch
import sys
import torch.nn as nn
from functools import partial
from einops import rearrange, reduce, repeat
from models.layers import MemoryEfficientAttention, SelfAttention
from models.lightning_trainer import LightningTrainer as LightningTrainerLUAR
from utilities.file_utils import Utils as utils
from transformers import AutoModel, RobertaTokenizer, AutoTokenizer  # Ensure tokenizer is available

class Transformer(LightningTrainerLUAR):
    """Defines a Transformer model for author style representation."""
    
    def __init__(self, params, tokenizer=None):
        super(Transformer, self).__init__(params)
        self.save_hyperparameters()

        self.create_transformer()

        self.learning_rate = params.learning_rate
        self.attn_fn = SelfAttention()
        self.linear = nn.Linear(self.hidden_size, self.params.embedding_dim)
        
        # Define classification head and loss function
        self.classifier = nn.Linear(self.params.embedding_dim, self.params.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()  # For classification tasks

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

    def get_episode_embeddings(self, data, chunk_size=128):
        """Computes author episode and comment-level embeddings in chunks to reduce memory load."""
        input_ids, attention_mask = data

        #print(f"input_ids shape: {input_ids.shape}")
        #print(f"attention_mask shape: {attention_mask.shape}")

        B, N, E, L = input_ids.shape
        assert input_ids.shape == attention_mask.shape, "Mismatch before flattening"

        # Flatten to [B * N * E, L]
        flattened_input_ids = input_ids.view(B * N * E, L)
        flattened_attention_mask = attention_mask.view(B * N * E, L)

        all_comment_embeddings = []

        # Loop over full flattened input
        for start_idx in range(0, flattened_input_ids.size(0), chunk_size):
            end_idx = min(start_idx + chunk_size, flattened_input_ids.size(0))

            chunk_input_ids = flattened_input_ids[start_idx:end_idx]
            chunk_attention_mask = flattened_attention_mask[start_idx:end_idx]

            outputs = self.transformer(chunk_input_ids, attention_mask=chunk_attention_mask)
            embeddings = outputs.last_hidden_state  # [chunk_size, L, hidden_size]

            pooled_embeddings = self.mean_pooling(embeddings, chunk_attention_mask)  # [chunk_size, hidden_size]
            all_comment_embeddings.append(pooled_embeddings)

            torch.cuda.empty_cache()

        # Concatenate all pooled outputs
        comment_embeddings = torch.cat(all_comment_embeddings, dim=0)  # [B*N*E, hidden_size]

        if comment_embeddings.numel() != B * N * E * self.hidden_size:
            raise ValueError(f"Invalid reshape: B={B}, N={N}, E={E}, hidden={self.hidden_size}, total={comment_embeddings.numel()}")

        # Reshape to [B, N, E, hidden_size]
        comment_embeddings = comment_embeddings.view(B, N, E, self.hidden_size)

        # Attention over [B, N, E, hidden_size]
        episode_embeddings = self.attn_fn(comment_embeddings, comment_embeddings, comment_embeddings)
        episode_embeddings = episode_embeddings.max(dim=2).values  # [B, N, hidden_size]
        episode_embeddings = reduce(episode_embeddings, 'b n h -> b h', 'max')  # [B, hidden_size]
        episode_embeddings = self.linear(episode_embeddings)  # Project to output dim

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

    def _model_forward(self, batch):
        #print(">>> _model_forward called", flush=True)
        """Passes a batch through the model."""
        (input_ids, attention_mask), labels = batch

        # Now `input_ids` and `attention_mask` are tensors, so no need for dictionary access
        # Forward pass through the encoder
        episode_embeddings, comment_embeddings = self.forward((input_ids, attention_mask))
        
        return episode_embeddings, comment_embeddings
    
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

