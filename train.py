import time
import math
import torch
import torch.nn as nn
from model import CharRNN
from generate import generate_text



def train_step(model, optimizer, criterion, dataset, chunk_len, batch_size, device):
    """Single training step (Vectorized/Parallelized version)."""
    model.train()
    inp, target = dataset.get_train_batch(chunk_len, batch_size, device)
    hidden = model.init_hidden(batch_size, device)

    optimizer.zero_grad()

    
    outputs, hidden = model(inp, hidden)

    loss = criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))

    # Detach hidden state to prevent backprop through entire history
    if isinstance(hidden, tuple):  # LSTM
        hidden = (hidden[0].detach(), hidden[1].detach())
    else:
        hidden = hidden.detach()

    loss.backward()

    # Gradient clipping to prevent exploding gradients
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return loss.item()


def validate(model, criterion, dataset, chunk_len, batch_size, device, n_batches=10):
    """Compute validation loss and perplexity (Vectorized/Parallelized version)."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(n_batches):
            inp, target = dataset.get_val_batch(chunk_len, batch_size, device)
            hidden = model.init_hidden(batch_size, device)
            
        
            outputs, hidden = model(inp, hidden)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))
            
            total_loss += loss.item()
            
    avg_loss = total_loss / n_batches
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity


def train_model(config, dataset, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    model = CharRNN(
        input_size=dataset.n_characters,
        hidden_size=config['hidden_size'],
        output_size=dataset.n_characters,
        model_type=config['model_type'],
        n_layers=config['n_layers'],
        dropout=config.get('dropout', 0.0)
    ).to(device)

    if verbose:
        print(f"Model parameters: {model.count_parameters():,}")

    opt_type = config.get('optimizer_type', 'adam').lower()
    lr = config['learning_rate']
    if opt_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'val_loss': [], 'val_perplexity': [],
        'samples': [], 'config': config, 'n_params': model.count_parameters()
    }

    chunk_len = config.get('chunk_len', 200)
    batch_size = config.get('batch_size', 64)
    n_epochs = config['n_epochs']
    print_every = config.get('print_every', 100)

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        loss = train_step(model, optimizer, criterion, dataset,
                          chunk_len, batch_size, device)

        if epoch % print_every == 0 or epoch == 1:
            val_loss, val_ppl = validate(model, criterion, dataset,
                                         chunk_len, batch_size, device)
            history['train_loss'].append((epoch, loss))
            history['val_loss'].append((epoch, val_loss))
            history['val_perplexity'].append((epoch, val_ppl))

            sample = generate_text(model, dataset, prime_str="The ",
                                   predict_len=100, temperature=0.8, device=device)
            history['samples'].append((epoch, sample))

            if verbose:
                elapsed = time.time() - start_time
                print(f"[{epoch}/{n_epochs}] ({elapsed:.0f}s) "
                      f"train_loss={loss:.4f} val_loss={val_loss:.4f} "
                      f"val_ppl={val_ppl:.2f}")
                print(f"  Sample: {sample[:80]}...")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if config.get('save_path'):
                    torch.save({
                        'model_state': model.state_dict(),
                        'config': config,
                        'chars': dataset.chars,
                        'char2idx': dataset.char2idx,
                        'idx2char': dataset.idx2char,
                    }, config['save_path'])

    total_time = time.time() - start_time
    history['total_time'] = total_time
    history['best_val_loss'] = best_val_loss
    history['best_val_ppl'] = math.exp(best_val_loss) if best_val_loss < 100 else float('inf')

    if verbose:
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Best val_loss: {best_val_loss:.4f}, Best val_ppl: {history['best_val_ppl']:.2f}")

    return history, model