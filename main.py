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
        # div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        div_term = 10000.0 ** (torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], :].requares_grad_(False)
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, image_size, patch_size, hidden_channels, num_classes, dropout):
        super().__init__()
        self.image_size = image_size
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.patch_embeddings = PatchEmbeddings(
            image_size, patch_size, hidden_channels, num_classes
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_channels))

        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, self.hidden_channels)
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

        # I know, I know :)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(q, k, v, dropout):
        d_k = q.shape[-1]
        scores = torch.matmul(q, k.T) / torch.sqrt(d_k)

        scores = torch.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        return torch.matmul(scores, v), scores

    def forward(self, q, k, v):
        batch_size = q.shape[0]

        q = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, attention_scores = MultiHeadAttention.attention(q, k, v, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(x)


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
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps) + self.bias
        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)

        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x)
        )
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x):
        for layer in layers:
            x = layer(x)
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, image_size, hidden_size, num_classes, layers, dropout):
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.layers = layers
        self.dropout = nn.Dropout(dropout)

        self.patch_embeddings = Embeddings(
            image_size, patch_size, hidden_size, num_classes, dropout
        )

        self.encoder = Encoder(layers)

        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        embedding_output = self.patch_embeddings(x)
        encoder_output, all_attentions = self.encoder(
            embedding_output, output_attentions=output_attentions
        )
        logits = self.classifier(encoder_output[:, 0])
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
