import torch
from torch import nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import AutoModel, AutoTokenizer


def login_required():
    from dotenv import load_dotenv
    import os
    load_dotenv(".env")  # Load environment variables from .env file
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file. Please set it to your Hugging Face token.")
    from huggingface_hub import login
    login(token)  # Login to Hugging Face using the token
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    print("Logged in to Hugging Face successfully.")


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Query from time series, Key and Value from LLM
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, ts_features, llm_features):
        """
        ts_features: [batch_size, patch_num, d_model] - Features from time series branch
        llm_features: [batch_size, seq_len, d_model] - Features from LLM branch
        """
        batch_size = ts_features.shape[0]

        # Linear projections
        q = self.q_linear(ts_features).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(llm_features).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(llm_features).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Output projection
        output = self.out_proj(context)
        return output


class LLMEmbedding(nn.Module):
    def __init__(self, llm_model_name="meta-llama/Meta-Llama-3.1-8B", d_model=512, max_length=512, device="cuda"):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.d_model = d_model

        # Load tokenizer and model
        try:
            print(f"Loading model: {llm_model_name} " + "*" * 10)
            login_required()  # Ensure Hugging Face login is done before loading the model
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token
            self.llm_model = AutoModel.from_pretrained(llm_model_name,
                                                       torch_dtype=torch.float16,  # Use half precision
                                                       device_map="auto",  # Automatically manage model placement,
                                                       low_cpu_mem_usage=True)  # Reduce CPU memory usage

            # Freeze LLM parameters
            for param in self.llm_model.parameters():
                param.requires_grad = False

            # Get LLM embedding dimension
            self.llm_dim = self.llm_model.config.hidden_size

            # Projection layer to match d_model
            self.projection = nn.Linear(self.llm_dim, d_model)

            self.llm_available = True
        except Exception as e:
            print(f"Warning: Could not load LLM model {llm_model_name}. Using random embeddings instead.")
            print(f"Error: {e}")
            self.llm_available = False
            # Create a dummy embedding layer
            self.dummy_embedding = nn.Embedding(max_length, d_model)
            self.projection = nn.Identity()

    def forward(self, text_descriptions=None, precomputed_embeddings=None):
        """
        Process text descriptions through LLM and project to d_model dimension
        or use precomputed embeddings

        Args:
            text_descriptions: List of text descriptions or tensor, or None if using precomputed_embeddings
            precomputed_embeddings: Precomputed embeddings tensor [batch_size, embedding_dim] or None
        """
        # If precomputed embeddings are provided, use them
        if precomputed_embeddings is not None:
            batch_size = precomputed_embeddings.shape[0]

            # Check if embeddings need projection
            if precomputed_embeddings.shape[-1] != self.d_model:
                # Create a projection layer if needed
                if not hasattr(self, 'precomputed_projection'):
                    self.precomputed_projection = nn.Linear(
                        precomputed_embeddings.shape[-1], self.d_model
                    ).to(self.device)

                # Project embeddings to d_model
                embeddings = self.precomputed_projection(precomputed_embeddings)
            else:
                embeddings = precomputed_embeddings

            # Reshape if needed to match expected dimensions [batch_size, seq_len, d_model]
            if len(embeddings.shape) == 2:
                # Expand to create a sequence dimension
                embeddings = embeddings.unsqueeze(1).repeat(1, self.max_length, 1)

            return embeddings

        # Process text descriptions if no precomputed embeddings
        if self.llm_available:
            # Đảm bảo text_descriptions là list of strings
            if isinstance(text_descriptions, torch.Tensor):
                # Nếu là tensor, chuyển đổi về list
                text_descriptions = text_descriptions.cpu().tolist()

            # Nếu text_descriptions là list nhưng không phải list of strings
            if isinstance(text_descriptions, list):
                # Đảm bảo mỗi phần tử là chuỗi
                text_descriptions = [str(item) if not isinstance(item, str) else item
                                     for item in text_descriptions]
            elif not isinstance(text_descriptions, str):
                # Nếu không phải list hoặc string, chuyển thành string
                text_descriptions = str(text_descriptions)

            # Tokenize text
            inputs = self.tokenizer(text_descriptions, return_tensors="pt", padding=True,
                                    truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get LLM embeddings
            with torch.no_grad():
                outputs = self.llm_model(**inputs)

            # Use the last hidden state
            embeddings = outputs.last_hidden_state  # [batch_size, seq_len, llm_dim]

            # Project to d_model
            embeddings = self.projection(embeddings)  # [batch_size, seq_len, d_model]

            return embeddings
        else:
            # Fallback to a simpler embedding if LLM is not available
            batch_size = len(text_descriptions) if isinstance(text_descriptions, list) else 1
            # Return a dummy tensor of the right shape
            return torch.zeros(batch_size, 1, self.d_model, device=self.device)


class Model(nn.Module):
    def __init__(self, configs):
        """
        Model that combines LLM embeddings with time series data using cross-attention
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        padding = configs.stride

        # Whether to use pre-computed embeddings
        self.use_precomputed_embeddings = getattr(configs, 'use_precomputed_embeddings', False)

        # LLM embedding component
        self.llm_embedding = LLMEmbedding(
            llm_model_name=getattr(configs, 'llm_model_name', "meta-llama/Meta-Llama-3.1-8B"),
            d_model=configs.d_model,
            max_length=getattr(configs, 'max_length', 512),
            device=self.device
        )

        # Time series branch: patching and embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, padding,
                                              configs.dropout)

        # Time series branch: encoder
        self.ts_encoder = Encoder(
            encoder_layers_1=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention_mechanism=FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                          output_attention=configs.output_attention),
                        d_model=configs.d_model, n_heads=configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ], norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Cross-attention for fusion
        self.cross_attention = CrossAttention(
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            dropout=configs.dropout
        )

        # Prediction Head
        self.head_nf = configs.d_model * int(
            (configs.seq_len - configs.patch_len) / configs.stride + 2)  # number of patches
        self.head = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_descriptions=None, precomputed_embeddings=None, mask=None):
        """
        x_enc: Input time series [Batch, seq_len, n_vars]
        x_mark_enc: Time features for input [Batch, seq_len, n_features]
        x_dec: Decoder input [Batch, label_len+pred_len, n_vars]
        x_mark_dec: Time features for decoder [Batch, label_len+pred_len, n_features]
        text_descriptions: List of text descriptions [Batch]
        precomputed_embeddings: Pre-computed embeddings from LLM [Batch, embedding_dim]
        """
        # If text descriptions are not provided and no precomputed embeddings, create dummy ones
        batch_size = x_enc.shape[0]
        if text_descriptions is None and precomputed_embeddings is None:
            text_descriptions = [""] * batch_size

        # Process time series data
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()  # [Batch, 1, seq_len]
        x_enc = x_enc - means  # [Batch, seq_len, n_vars]
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev  # Change the scale to stabilize the training

        # Patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [Batch, n_vars, seq_len]
        ts_out, n_vars = self.patch_embedding(x_enc)  # [Batch*n_vars, patch_num, d_model]

        # Time series encoder
        ts_out, _ = self.ts_encoder(ts_out)  # [Batch*n_vars, patch_num, d_model]

        # Reshape time series features
        patch_num = ts_out.shape[1]
        ts_out = ts_out.view(batch_size, n_vars, patch_num, -1)  # [Batch, n_vars, patch_num, d_model]

        # Process each variable separately with cross-attention
        fused_features = []
        for i in range(n_vars):
            # Get time series features for this variable
            var_ts_features = ts_out[:, i]  # [Batch, patch_num, d_model]

            # Get LLM embeddings - either from text descriptions or pre-computed
            llm_features = self.llm_embedding(
                text_descriptions=text_descriptions,
                precomputed_embeddings=precomputed_embeddings
            )  # [Batch, seq_len, d_model]

            # Apply cross-attention
            fused_var = self.cross_attention(var_ts_features, llm_features)  # [Batch, patch_num, d_model]
            fused_features.append(fused_var.unsqueeze(1))  # [Batch, 1, patch_num, d_model]

        # Combine all variables
        fused_out = torch.cat(fused_features, dim=1)  # [Batch, n_vars, patch_num, d_model]

        # Permute for prediction head
        fused_out = fused_out.permute(0, 1, 3, 2)  # [Batch, n_vars, d_model, patch_num]

        # Prediction Head
        x = self.head(fused_out)  # [Batch, n_vars, pred_len]
        x = x.permute(0, 2, 1)  # [Batch, pred_len, n_vars]

        # De-normalization
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)) + \
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return x[:, -self.pred_len:, :]  # [Batch, pred_len, n_vars]
