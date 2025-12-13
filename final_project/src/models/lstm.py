"""
LSTM-based sign language classifier.

Baseline model: Takes sequence of hand landmarks and predicts sign class.
"""

import torch
import torch.nn as nn


class SignLSTM(nn.Module):
    """
    Two-layer LSTM for sign language classification.

    Architecture:
        Input (T, 63) -> LSTM -> FC -> Output (num_classes)
    """

    def __init__(
        self,
        input_dim: int = 63,      # 21 landmarks * 3 coords
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output dimension depends on bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Fully connected classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state for classification
        # For bidirectional: concatenate forward and backward final states
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden_dim)
            # Take last layer's forward and backward hidden states
            forward_h = h_n[-2, :, :]   # Last layer forward
            backward_h = h_n[-1, :, :]  # Last layer backward
            hidden = torch.cat([forward_h, backward_h], dim=1)
        else:
            hidden = h_n[-1, :, :]  # Last layer hidden state

        # Classification
        logits = self.fc(hidden)
        return logits


class SignLSTMWithAttention(nn.Module):
    """
    LSTM with attention mechanism for sign language classification.

    Attention helps focus on important frames in the sequence.
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = hidden_dim * 2

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Compute attention weights
        attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # Classification
        logits = self.fc(context)
        return logits
