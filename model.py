import torch
import torch.nn as nn


class PatchEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, hidden_channels):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = 10000.0 ** (torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, hidden_channels, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_channels = hidden_channels

        self.patch_embeddings = PatchEmbeddings(
            image_size, patch_size, in_channels, hidden_channels
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_channels))

        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, hidden_channels)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def attention(self, q, k, v):
        d_k = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )

        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        return output, scores

    def forward(self, q, k, v):
        batch_size = q.shape[0]

        q = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, attention_scores = self.attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(x), attention_scores


class MLP(nn.Module):
    def __init__(self, d_model, hidden_channels, dropout):
        super().__init__()
        self.d_model = d_model
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model=None, eps=1e-6):
        super().__init__()
        self.eps = eps
        if d_model is not None:
            self.alpha = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.alpha = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.alpha * x + self.bias


class EncoderBlock(nn.Module):
    def __init__(self, d_model, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, output_attentions=False):
        # Self attention
        norm_x = self.norm1(x)
        attn_output, attention_scores = self.self_attention_block(
            norm_x, norm_x, norm_x
        )
        x = x + self.dropout(attn_output)

        # Feed forward
        norm_x = self.norm2(x)
        ff_output = self.feed_forward_block(norm_x)
        x = x + self.dropout(ff_output)

        if output_attentions:
            return x, attention_scores
        return x


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNorm()

    def forward(self, x, output_attentions=False):
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            if output_attentions:
                x, attention_scores = layer(x, output_attentions=True)
                all_attentions.append(attention_scores)
            else:
                x = layer(x)

        x = self.norm(x)

        if output_attentions:
            return x, all_attentions
        return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        hidden_size,
        num_classes,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Create patch embeddings
        self.embeddings = Embeddings(
            image_size, patch_size, in_channels, hidden_size, dropout
        )

        # Create encoder blocks
        layers = []
        for _ in range(num_layers):
            attention = MultiHeadAttention(hidden_size, num_heads, dropout)
            mlp = MLP(hidden_size, mlp_dim, dropout)
            encoder_block = EncoderBlock(hidden_size, attention, mlp, dropout)
            layers.append(encoder_block)

        self.encoder = Encoder(layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize weights using standard Transformer weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, output_attentions=False):
        # Get embeddings
        embedding_output = self.embeddings(x)

        # Pass through encoder
        if output_attentions:
            encoder_output, all_attentions = self.encoder(
                embedding_output, output_attentions=True
            )
        else:
            encoder_output = self.encoder(embedding_output)
            all_attentions = None

        # Classification from CLS token
        logits = self.classifier(encoder_output[:, 0])

        if not output_attentions:
            return logits
        else:
            return logits, all_attentions


if __name__ == "__main__":
    # Create a sample input tensor [batch_size, channels, height, width]
    batch_size = 8
    channels = 3
    img_size = 224
    x = torch.randn(batch_size, channels, img_size, img_size)

    # Create ViT model
    model = ViT(
        image_size=img_size,
        patch_size=16,
        in_channels=channels,
        hidden_size=768,
        num_classes=1000,
        num_layers=12,
        num_heads=12,
    )

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
