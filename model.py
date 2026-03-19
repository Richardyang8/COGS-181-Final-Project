import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 model_type="lstm", n_layers=2, dropout=0.0):
        super(CharRNN, self).__init__()
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_rate = dropout

        # Encoder: character index -> embedding vector
        self.encoder = nn.Embedding(input_size, hidden_size)

        # RNN layer (supports rnn, lstm, gru)
        
        rnn_dropout = dropout if n_layers > 1 else 0.0
        if self.model_type == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers,
                              dropout=rnn_dropout, batch_first=True)
        elif self.model_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=rnn_dropout, batch_first=True)
        elif self.model_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers,
                               dropout=rnn_dropout, batch_first=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'rnn', 'lstm', or 'gru'.")

        # Dropout before decoder
        self.dropout = nn.Dropout(dropout)

        # Decoder: hidden state -> character probabilities
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        """
        input: (batch_size,) for generation OR (batch_size, chunk_len) for training
        hidden: hidden state from previous step
        """
        
        if input.dim() == 1:
           
            input = input.unsqueeze(1)
            is_single_step = True
        else:
        
            is_single_step = False

       
        encoded = self.encoder(input)  
        
       
        output, hidden = self.rnn(encoded, hidden)
        output = self.dropout(output)
        
       
       
        logits = self.decoder(output)  
        
        if is_single_step:
           
            logits = logits.squeeze(1)

        return logits, hidden

    def init_hidden(self, batch_size, device='cpu'):
        """Initialize hidden state"""
        if self.model_type == "lstm":
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))
        else:
            return torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)