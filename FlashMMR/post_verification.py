import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class PostVerificationRNN(nn.Module):
    """
    Aggregate variable-length candidate features with an RNN and score them with an MLP.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(hidden_dim // 4, 1)  # produce a scalar score
        )

    def forward(self, interval_features_list):
        """
        interval_features_list: List[Tensor],
            e.g., a list of length N where each element has shape (L_i, d) for one interval.
        Returns: scores with shape (N, 1).
        """
        lengths = [f.shape[0] for f in interval_features_list]
        N = len(lengths)

        sorted_lengths, sorted_idx = torch.sort(torch.tensor(lengths), descending=True)
        sorted_features = [interval_features_list[i] for i in sorted_idx]

        packed_input = rnn_utils.pack_sequence(sorted_features, enforce_sorted=False)

        packed_output, hidden = self.rnn(packed_input)

        hidden = hidden[-1]  # (N, hidden_dim)

        _, unsorted_idx = torch.sort(sorted_idx)
        hidden = hidden[unsorted_idx]  # (N, hidden_dim)
        
        scores = self.classifier(hidden)  # (N, 1)
        return scores


if __name__ == "__main__":
    interval_features_list = [
        torch.randn(10, 256),
        torch.randn(12, 256),
        torch.randn(7, 256),
        torch.randn(20, 256),
        torch.randn(15, 256)
    ]

    model = PostVerificationRNN(input_dim=256, hidden_dim=128)
    scores = model(interval_features_list)
    print(scores.shape) 