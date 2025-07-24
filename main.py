import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from pathlib import Path
from pickle import dump as pkl_dump, load as pkl_load
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_VAL_SPLIT = 0.9

BLOCK_SIZE = 8
EMBED_DIM = 36
N_HEADS = 4
N_LAYERS = 6
DROPOUT = 0.15

BATCH_SIZE = 64
LR = 1e-4
MAX_EPOCHS = 100
PATIENCE = 5

SPECIAL_TOKENS = {
    'START_TOKEN': '<START>',
    'PAD_TOKEN': '<PAD>',
    'END_TOKEN': '<END>',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    raise RuntimeError("This script requires a GPU to run efficiently.")

BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "_datasets"
FULLDATASET_PATH = DATASETS_DIR / "full_dataset.pkl"
CHECKPOINT_PATH = BASE_DIR / "model.pt"
VOCAB_PATH = BASE_DIR / "vocab.json"

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_vocab(names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    chars: list[str] = sorted(set(''.join(names)))
    stoi: dict[str, int] = {ch: i for i, ch in enumerate(SPECIAL_TOKENS.values(), start=0)}
    for i, ch in enumerate(chars, start=len(stoi)):
        stoi[ch] = i
    itos: dict[int, str] = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def save_vocab(stoi: dict[str, int], itos: dict[int, str], path: Path):
    """Save vocabulary for later use"""
    vocab_data = {'stoi': stoi, 'itos': itos}
    with open(path, 'w') as f:
        json.dump(vocab_data, f)

def load_vocab(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load vocabulary from file"""
    with open(path, 'r') as f:
        vocab_data = json.load(f)
    # Convert string keys back to integers for itos
    itos = {int(k): v for k, v in vocab_data['itos'].items()}
    return vocab_data['stoi'], itos

def encode(s: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[SPECIAL_TOKENS['START_TOKEN']]] + [stoi[c] for c in s] + [stoi[SPECIAL_TOKENS['END_TOKEN']]]

def decode(toks: list[int], itos: dict[int, str], stoi: dict[str, int]) -> str:
    #special_token_ids = {stoi[tok] for tok in SPECIAL_TOKENS.values()}
    return ''.join(itos[i] for i in toks) #if i not in special_token_ids)

def check_health(x: torch.Tensor, name: str):
    """Check tensor for NaN or Inf values"""
    if torch.isnan(x).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(x).any():
        raise ValueError(f"Inf detected in {name}")
    if x.numel() == 0:
        raise ValueError(f"Empty tensor detected in {name}")
    if x.dim() < 1:
        raise ValueError(f"Tensor {name} has less than 1 dimension")


class NameDataset(Dataset):
    def __init__(self, words: list[str], stoi: dict[str, int]) -> None:
        X: list[list[int]] = []
        y: list[int] = []
        for w in words:
            context: list[int] = [stoi[SPECIAL_TOKENS['PAD_TOKEN']]] * BLOCK_SIZE
            chr_is: list[int] = encode(w, stoi) # Encode with START and END tokens
            for i in chr_is:
                X.append(context.copy())
                y.append(i)
                context = context[1:] + [i]
        
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class NameTransformer(nn.Module):
    def __init__(self, stoi: dict[str, int], itos: dict[int, str]) -> None:
        super().__init__()

        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

        self.token_embedding = nn.Embedding(self.vocab_size, EMBED_DIM, padding_idx=stoi[SPECIAL_TOKENS['PAD_TOKEN']])
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        
        # Add layer normalization and dropout
        self.embedding_dropout = nn.Dropout(DROPOUT)
        self.embedding_norm = nn.LayerNorm(EMBED_DIM)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=N_HEADS,
            dim_feedforward=4 * EMBED_DIM,
            dropout=DROPOUT,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=N_LAYERS)
        
        self.output_norm = nn.LayerNorm(EMBED_DIM)
        self.project = nn.Linear(EMBED_DIM, self.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, causal_mask=None, pos=None) -> torch.Tensor:
        B, T = idx.shape
        assert T <= BLOCK_SIZE, f"Sequence length {T} exceeds block size {BLOCK_SIZE}"

        if pos is None:
            pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        
        # Embeddings with normalization and dropout
        x = self.token_embedding(idx) + self.pos_embedding(pos)
        check_health(x, "Token and position embeddings")
        x = self.embedding_norm(x)
        check_health(x, "After embedding normalization")
        x = self.embedding_dropout(x)
        check_health(x, "After embedding dropout")

        if causal_mask is None:
            # causal_mask = (torch.triu(torch.ones(T, T), diagonal=1) == 1)\
            #     .transpose(0, 1)\
            #     .float()\
            #     .masked_fill(
            #         torch.triu(torch.ones(T, T), diagonal=1) == 1, float('-inf')
            #     )\
            #     .masked_fill(
            #         torch.triu(torch.ones(T, T), diagonal=1) == 0, float(0.0)
            #     ).to(idx.device)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)

        x = self.transformer(x, causal_mask)
        check_health(x, "After transformer")
        x = self.output_norm(x)
        check_health(x, "After output normalization")
        x = self.project(x)
        check_health(x, "After projection")

        return x

    @torch.no_grad()
    def generate_single_word(
        self,
        max_new_tokens: int = 32,
        temperature: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.9
    ) -> str:
        """Generate a single name with improved sampling"""
        def top_k_top_p_filtering(logits, top_k=0, top_p=0.9):
            # """Filter logits using top-k and/or nucleus (top-p) sampling"""
            # if top_k > 0:
            #     # Remove tokens with rank > top_k
            #     indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
            #     logits[indices_to_remove] = float('-inf')
            
            # if top_p > 0.0:
            #     # Sort logits and compute cumulative probabilities
            #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
            #     # Remove tokens with cumulative probability above threshold
            #     sorted_indices_to_remove = cumulative_probs > top_p
            #     sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            #     sorted_indices_to_remove[:, 0] = 0
                
            #     # Scatter back to original indexing
            #     indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            #     logits[indices_to_remove] = float('-inf')
            
            return logits

        self.eval()
        
        end_tok_id = self.stoi[SPECIAL_TOKENS['END_TOKEN']]
        pad_tok_id = self.stoi[SPECIAL_TOKENS['PAD_TOKEN']]
        context = [pad_tok_id] * BLOCK_SIZE
        out_word = []

        for _ in range(max_new_tokens):
            model_input = torch.tensor([context], dtype=torch.long, device=DEVICE)
            logits = self(model_input)
            logits = logits[:, -1, :] / temperature
            
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            
            probs = F.softmax(logits, dim=-1)
            next_tok_id = torch.multinomial(probs, num_samples=1).item()
            if next_tok_id == end_tok_id:
                break

            out_word.append(next_tok_id)
            context = context[1:] + [next_tok_id]

        return decode(out_word, self.itos, self.stoi)


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """Enhanced checkpoint saving with more metadata"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'model_config': {
            'stoi': model.stoi,
            'embed_dim': EMBED_DIM,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'block_size': BLOCK_SIZE,
            'dropout': DROPOUT
        }
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load checkpoint with error handling"""
    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('val_loss', float('inf'))
    except FileNotFoundError:
        logger.info(f"No checkpoint found.")
        return 0, float('inf')


def run_epoch_and_get_avg_loss(model, dataloader, vocab_size, optimizer=None, is_train=True) -> float:
    """Run a single epoch and return average loss"""
    if is_train:
        model.train()
    else:
        assert optimizer is None, "Optimizer should not be provided for validation"
        model.eval()

    loop = tqdm(dataloader, desc='train' if is_train else 'val', leave=False)
    total_loss = 0.0
    num_tokens = 0

    for xb, yb in loop:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        with torch.no_grad() if not is_train else contextlib.nullcontext():
            logits = model(xb)
            check_health(logits, "Logits")
            logits_for_loss = logits[:, -1, :]  # returns (batch_size, vocab_size), where
            # the vocab_dims correspond to the embedding dims of the last token
            loss = F.cross_entropy(logits_for_loss, yb)

            if is_train:
                optimizer.zero_grad() # type: ignore
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                if optimizer is not None:
                    optimizer.step()
        
        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        num_tokens += batch_size

        loop.set_postfix(loss=loss.item())
    
    if num_tokens == 0:
        logger.warning("No valid tokens processed, returning zero loss")
        return 0.0
    
    return total_loss / num_tokens

def train(model, train_loader, val_loader, vocab_size, optimizer, scheduler, generate_names=False):
    """Training loop with early stopping"""

    start_epoch, best_val_loss = load_checkpoint(model, optimizer, CHECKPOINT_PATH)
    if start_epoch > 0:
        logger.info(f"Resumed from epoch {start_epoch}")

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, MAX_EPOCHS):
        for i in range(5):
            name = model.generate_single_word(temperature=0.8, top_k=20, top_p=0.9)
            logger.info(f"{i+1:2d}: {name}")
        
        avg_train_loss = run_epoch_and_get_avg_loss(
            model, train_loader, vocab_size, optimizer, is_train=True
        )
        avg_val_loss = run_epoch_and_get_avg_loss(
            model, val_loader, vocab_size, is_train=False
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch+1}/{MAX_EPOCHS} | "
                   f"Train Loss: {avg_train_loss:.4f} | "
                   f"Val Loss: {avg_val_loss:.4f} | "
                   f"LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, avg_val_loss, CHECKPOINT_PATH)
            logger.info(f"New best model saved (Val Loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

def load_names() -> list[str]:
    """Load names from dataset files or full dataset cache"""

    if FULLDATASET_PATH.exists():
        logger.info(f"Loading full dataset from {FULLDATASET_PATH}")
        with open(FULLDATASET_PATH, 'rb') as f:
            return pkl_load(f)
        
    all_names: list[str] = []
    dataset_files: list[Path] = list(DATASETS_DIR.glob('*.txt'))

    if not dataset_files:
        logger.error(f"No dataset files found in {DATASETS_DIR}")
        return []
    
    for dataset_file in dataset_files:
        logger.info(f"Loading dataset: {dataset_file.name}")
        with open(dataset_file, "r", encoding='utf-8') as f:
            names = [line.strip().lower() for line in f if line.strip()]
            all_names.extend(names)
            logger.info(f"  Loaded {len(names)} names")
    
    with open(FULLDATASET_PATH, 'wb') as f:
        pkl_dump(all_names, f)

    return all_names

def main():
    all_names: list[str] = load_names()
    
    # Build and save vocabulary
    stoi, itos = build_vocab(all_names)
    vocab_size = len(stoi)
    
    train_dataset, val_dataset = random_split(NameDataset(all_names, stoi), [TRAIN_VAL_SPLIT, 1 - TRAIN_VAL_SPLIT])
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        drop_last=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        drop_last=False, num_workers=4, pin_memory=True
    )
    
    model = NameTransformer(stoi, itos).to(DEVICE)
    
    # Optimizer with smaller weight decay for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.001)
    # Learning rate scheduler
    warmup_epochs = 2
    main_epochs = MAX_EPOCHS - warmup_epochs
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=LR * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    logger.info("Starting training...")
    train(model, train_loader, val_loader, vocab_size, optimizer, scheduler, generate_names=True)
    
    logger.info("Loading best model for generation...")
    load_checkpoint(model, None, CHECKPOINT_PATH)
    model.eval()
    
    logger.info("\nGenerated names:")
    for i in range(20):
        name = model.generate_single_word()
        logger.info(f"{i+1:2d}: {name}")


if __name__ == "__main__":
    main()