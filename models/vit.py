import torch
import torch.nn as nn
import torch.nn.functional as F

class VITConfig:
    def __init__(
        self,
        hidden_dim=384,
        num_hidden_layers=5,
        num_attention_heads=8,
        num_channels=3,
        image_size=64,
        patch_size=2,
        mlp_dim=512, # Added this
        drop_rate=0.1, # Added this
        **kwargs):
        
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.mlp_dim = mlp_dim
        self.drop_rate = drop_rate


class VARCMultiHeadAttention(nn.Module):
    def __init__(self, config: VITConfig):
        super().__init__()
        self.config = config
        self.embed_size = self.config.hidden_dim
        self.head_dim = self.embed_size // self.config.num_attention_heads
        self.num_heads = self.config.num_attention_heads
        self.scale = (self.head_dim)**-0.5
        
        self.Q_matrix = nn.Linear(self.embed_size, self.embed_size)
        self.K_matrix = nn.Linear(self.embed_size, self.embed_size)
        self.V_matrix = nn.Linear(self.embed_size, self.embed_size)
        
        self.out_proj = nn.Linear(self.embed_size, self.embed_size)
        
    def forward(self, patch_embeds, attention_mask):
        #[B, Num_Patches, Embed_dim]
        batch_size, seq_length, embed_dim = patch_embeds.shape()
        #[B, Num_Patches, Embed_dim] X [Embed_dim, Embed_dim] --> [B, Num_Patches, Embed_dim]
        queries = self.Q_matrix(patch_embeds)
        # [B, Num_Patches, Embed_dim]
        keys = self.K_matrix(patch_embeds)
        # [B, Num_Patches, Embed_dim]
        values = self.V_matrix(patch_embeds)

        # [B, Num_Patches, Embed_dim] --> # [B, Num_Heads, Num_Patches, Head_Dim]
        queries = queries.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        # [B, Num_Patches, Embed_dim] --> # [B, Num_Heads, Num_Patches, Head_Dim]
        keys = keys.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        # [B, Num_Patches, Embed_dim] --> # [B, Num_Heads, Num_Patches, Head_Dim]
        values = values.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        # [B, Num_Heads, Num_Patches, Head_Dim] X # [B, Num_Heads, Head_Dim, Num_Patches] --> [B, Num_Heads, Num_Patches, Num_Patches] 
        attn_scores = queries @ keys.transpose(2,3)
        # divide by sqrt of head dim
        attn_scores = attn_scores * self.scale
        # apply masking to BG tokens
        if attention_mask is not None:
            # [B, Num_Patches] -->  [B, 1, 1, Num_Patches]
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(expanded_mask, float('-inf'))
        # [B, Num_Heads, Num_Patches, Num_Patches] 
        attn_weights = F.softmax(attn_scores, dim=-1)
        # [B, Num_Heads, Num_Patches, Num_Patches] X [B, Num_Heads, Num_Patches, Head_dim] --> [B, Num_Patches, Num_Heads, Head_dim] 
        attn_output = (attn_weights @ values).transpose(1,2).contiguous()
        # [B, Num_Patches, Embed_Dim] 
        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim)
        # [B, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)
        return self


class VARCMlp(nn.Module):
    def __init__(self, config: VITConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_dim)
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VARCEncoderBlock(nn.Module):
    def __init__(self, config: VITConfig):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(self.config.hidden_dim)
        self.attn = VARCMultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(self.config.hidden_dim)
        self.mlp = VARCMlp(config)

    def forward(self, patch_embeddings, attention_mask):
        #[B, Num_Patches, Embed_dim]
        shortcut = patch_embeddings
        patch_embeddings = self.norm1(patch_embeddings)
        patch_embeddings = self.attn(patch_embeddings, attention_mask)
        patch_embeddings += shortcut 
        shortcut = patch_embeddings
        patch_embeddings = self.norm2(patch_embeddings)
        patch_embeddings = self.mlp(patch_embeddings)
        patch_embeddings += shortcut
        return patch_embeddings


class VARCEncoder(nn.Module):
    def __init__(self, config: VITConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            VARCEncoderBlock(VITConfig) for _ in range(self.config.num_hidden_layers)
        ])
        
    def forward(self, hidden_states, attention_mask):
        for block in self.layers:
            hidden_states = block(hidden_states, attention_mask)
        return hidden_states


class VARCSeparablePositionEmbedding(nn.Module):
    def __init__(self, config: VITConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_dim
        self.grid_size = config.image_size // config.patch_size 
        
        # NOTE: If inputs are raw RGB integers, Conv2d works nicely as a projection
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, 
                                         out_channels=self.embed_dim, 
                                         kernel_size=config.patch_size, 
                                         stride=config.patch_size)
        
        self.half_dim = self.embed_dim // 2 
        self.x_embed = nn.Embedding(self.grid_size, self.half_dim)
        self.y_embed = nn.Embedding(self.grid_size, self.half_dim)
        self.bg_value = 10 

    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        device = pixel_values.device

        # 1. mask: [B, C*P*P, Num_Patches]
        raw_patches = torch.nn.functional.unfold(
            pixel_values, 
            kernel_size=self.config.patch_size, 
            stride=self.config.patch_size
        )
        
        # check if all pixels in patch == bg
        is_bg = (raw_patches == self.bg_value).all(dim=1) # [B, Num_Patches]
        
        # get patch Embeddings
        patches = self.patch_embedding(pixel_values) # [B, Dim, H/P, W/P]
        patches = patches.flatten(2).transpose(1, 2) # [B, Num_Patches, Dim]

        # add 2D positional Embeddings
        ids = torch.arange(self.grid_size, device=device)
        y_grid, x_grid = torch.meshgrid(ids, ids, indexing='ij')
        
        y_vecs = self.y_embed(y_grid.flatten()) # [Num_Patches, Half]
        x_vecs = self.x_embed(x_grid.flatten()) # [Num_Patches, Half]
        pos_embed = torch.cat([x_vecs, y_vecs], dim=-1) # [Num_Patches, Dim]

        embeddings = patches + pos_embed.unsqueeze(0)
        
        return embeddings, is_bg



class VARCModel(nn.Module):
    def __init__(self, config: VITConfig):
        super().__init__()
        self.config = config
        
        # get embeddings
        self.embeddings = VARCSeparablePositionEmbedding(config)
        
        # buil task token library for 400 training tasks
        self.task_embed = nn.Embedding(400, config.hidden_dim) 
        
        # encoder
        self.blocks = nn.ModuleList([
            VARCEncoderBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.norm_final = nn.LayerNorm(config.hidden_dim)
        
        # predictor head (outputs 11 classes: 0-9 + bg)
        self.classifier = nn.Linear(config.hidden_dim, 11) 

    def forward(self, pixel_values, task_ids):
        # pixel_values: [B, C, H, W]
        # task_ids: [B]
        
        # embed Images
        x, bg_mask = self.embeddings(pixel_values) # x: [B, 1024, Dim]
        
        # insert Task Token
        task_token = self.task_embed(task_ids).unsqueeze(1) # [B, 1, Dim]
        x = torch.cat([task_token, x], dim=1) # [B, 1025, Dim]
        
        # adjust mask (dont mask task token), prepend 'False' to the mask
        task_mask = torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.bool)
        full_mask = torch.cat([task_mask, bg_mask], dim=1) # [B, 1025]
        
        # transformer goes brrr...
        for block in self.blocks:
            x = block(x, full_mask)
            
        x = self.norm_final(x)
        
        # discard task token, only image tokens needed
        # [B, 1024, 11]
        image_tokens = x[:, 1:, :] 
        logits = self.classifier(image_tokens) 
        
        # for now, returning [B, 1024, 11]. --> will need reshaping for loss calculations. # todo. 
        return logits
