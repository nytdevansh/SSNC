"""
Utility classes and functions for the SSNC projects.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class SimpleTextEncoder(nn.Module):
    """Lightweight text encoder with dropout."""
    
    def __init__(self, embedding_dim=256, max_seq_length=100, dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Character-level encoding
        self.char_embed = nn.Embedding(256, 32)
        self.dropout1 = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embedding_dim)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, texts):
        """Encode batch of text strings."""
        batch_encodings = []
        
        for text in texts:
            char_ids = torch.tensor(
                [ord(c) % 256 for c in text[:self.max_seq_length]], 
                dtype=torch.long, 
                device=self.char_embed.weight.device
            )
            
            if len(char_ids) == 0:
                char_ids = torch.zeros(1, dtype=torch.long, device=self.char_embed.weight.device)
            
            # Embed and encode with dropout
            x = self.char_embed(char_ids)
            x = self.dropout1(x)
            x = x.transpose(0, 1).unsqueeze(0)
            x = torch.relu(self.conv1(x))
            x = self.dropout2(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            x = self.fc(x)
            x = self.dropout3(x)
            
            batch_encodings.append(x)
        
        return torch.cat(batch_encodings, dim=0)


class WordLevelEncoder(nn.Module):
    """
    Word-level encoder with learned embeddings and bidirectional LSTM.
    
    Much better than character-level for semantic understanding of questions.
    Uses a simple vocabulary built from training data.
    """
    
    def __init__(self, embedding_dim=256, max_seq_length=50, dropout=0.3, 
                 vocab_size=10000, word_embed_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        # Word vocabulary (built during first forward pass or explicitly)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.vocab_built = False
        
        # Word embedding
        self.word_embed = nn.Embedding(vocab_size, word_embed_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout)
        
        # Bidirectional LSTM for context
        self.lstm = nn.LSTM(
            word_embed_dim, 
            embedding_dim // 2,  # Bidirectional doubles this
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Attention pooling
        self.attention = nn.Linear(embedding_dim, 1)
        
        # Output projection
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def tokenize(self, text):
        """Simple word tokenization."""
        import re
        # Lowercase and split on non-alphanumeric
        words = re.findall(r'\b\w+\b', text.lower())
        return words[:self.max_seq_length]
    
    def build_vocab(self, texts, max_vocab=None):
        """Build vocabulary from training texts."""
        from collections import Counter
        
        word_counts = Counter()
        for text in texts:
            words = self.tokenize(text)
            word_counts.update(words)
        
        max_vocab = max_vocab or (self.vocab_size - 2)  # Reserve PAD and UNK
        most_common = word_counts.most_common(max_vocab)
        
        for word, _ in most_common:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        self.vocab_built = True
        print(f"✓ Built vocabulary: {len(self.word_to_idx)} words")
    
    def encode_text(self, text):
        """Convert text to word indices."""
        words = self.tokenize(text)
        indices = [self.word_to_idx.get(w, 1) for w in words]  # 1 = UNK
        return indices
    
    def forward(self, texts):
        """Encode batch of text strings."""
        device = self.word_embed.weight.device
        
        # Encode all texts
        batch_indices = []
        max_len = 0
        
        for text in texts:
            indices = self.encode_text(text)
            if len(indices) == 0:
                indices = [1]  # UNK if empty
            batch_indices.append(indices)
            max_len = max(max_len, len(indices))
        
        # Pad to same length
        padded = []
        for indices in batch_indices:
            padded.append(indices + [0] * (max_len - len(indices)))
        
        # Convert to tensor
        x = torch.tensor(padded, dtype=torch.long, device=device)
        
        # Embed words
        x = self.word_embed(x)  # [batch, seq_len, word_embed_dim]
        x = self.dropout1(x)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, embedding_dim]
        
        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        x = (lstm_out * attn_weights).sum(dim=1)  # [batch, embedding_dim]
        
        # Output projection
        x = self.fc(x)
        x = self.dropout2(x)
        x = self.layer_norm(x)
        
        return x


class TFIDFEncoder(nn.Module):
    """
    TF-IDF based encoder - often works BETTER than neural encoders for short text.
    
    Uses sklearn's TfidfVectorizer internally, projects to embedding_dim.
    """
    
    def __init__(self, embedding_dim=256, max_features=5000, dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        self._dropout_rate = dropout
        
        # Will be fitted during first training
        self.vectorizer = None
        self.fitted = False
        self.actual_features = max_features  # Will be updated after fit
        
        # Initialize projection layers NOW (not in fit) so they're properly registered!
        # This is critical - layers created in fit() aren't tracked by optimizer
        self.projection = nn.Linear(max_features, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def fit(self, texts):
        """Fit TF-IDF vectorizer on training texts."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english',
            min_df=2
        )
        self.vectorizer.fit(texts)
        
        self.actual_features = len(self.vectorizer.vocabulary_)
        print(f"✓ TF-IDF fitted: {self.actual_features} features (max: {self.max_features})")
        
        # If actual features < max_features, we need to handle padding in forward()
        # The projection layer is already initialized with max_features
        
        self.fitted = True
    
    def forward(self, texts):
        """Encode batch of text strings."""
        if not self.fitted:
            raise RuntimeError("TFIDFEncoder must be fitted before use. Call .fit(texts) first.")
        
        # Get TF-IDF features
        tfidf_matrix = self.vectorizer.transform(texts).toarray()
        
        # Convert to tensor
        device = self.projection.weight.device
        x = torch.tensor(tfidf_matrix, dtype=torch.float32, device=device)
        
        # Pad to max_features if needed (in case actual_features < max_features)
        if x.shape[1] < self.max_features:
            padding = torch.zeros(x.shape[0], self.max_features - x.shape[1], device=device)
            x = torch.cat([x, padding], dim=1)
        
        # Project to embedding_dim
        x = self.projection(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        
        return x


class GloVeEncoder(WordLevelEncoder):
    """
    Word-level encoder initialized with pre-trained GloVe embeddings.
    Extends WordLevelEncoder but replaces random embeddings with GloVe.
    """
    
    def __init__(self, embedding_dim=256, max_seq_length=50, dropout=0.3, 
                 vocab_size=10000, glove_path=None):
        # Initialize with word_embed_dim=100 (standard GloVe size)
        super().__init__(
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
            vocab_size=vocab_size,
            word_embed_dim=100  # Fixed for GloVe 100d
        )
        self.glove_path = glove_path
        
    def build_vocab(self, texts, max_vocab=None):
        """Build vocab and load GloVe embeddings."""
        # 1. Build standard vocabulary from texts
        super().build_vocab(texts, max_vocab)
        
        if not self.glove_path:
            print("⚠️ No GloVe path provided, using random embeddings")
            return
            
        print(f"Loading GloVe embeddings from {self.glove_path}...")
        
        # 2. Load GloVe vectors
        glove_vectors = {}
        try:
            with open(self.glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    glove_vectors[word] = vector
        except Exception as e:
            print(f"❌ Error loading GloVe: {e}")
            return
            
        print(f"✓ Loaded {len(glove_vectors)} GloVe vectors")
        
        # 3. Create embedding matrix
        hits = 0
        misses = 0
        embedding_matrix = np.zeros((len(self.word_to_idx), 100))
        
        # Initialize with random weights (for unknown words)
        embedding_matrix = np.random.normal(scale=0.6, size=(len(self.word_to_idx), 100))
        embedding_matrix[0] = np.zeros(100)  # PAD is zero
        
        for word, idx in self.word_to_idx.items():
            if word in glove_vectors:
                embedding_matrix[idx] = glove_vectors[word]
                hits += 1
            else:
                misses += 1
                
        # 4. Load into embedding layer
        self.word_embed = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False,  # Fine-tune embeddings
            padding_idx=0
        )
        # Re-register to correct device if needed
        # (will be handled when model moves to device)
        
        print(f"✓ Embeddings initialized: {hits} hits, {misses} misses ({hits/(hits+misses):.1%} coverage)")


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for regularization."""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, target):
        log_probs = torch.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def visualize_network_architecture(model, output_dir, epoch=None):
    """
    Visualize the SSNC network architecture with neurons and weights.
    
    Shows:
    - Layer structure with neuron counts
    - Neuron states (active, soft-deleted, marked)
    - Weight magnitude distributions
    - Utility distributions per layer
    """
    if hasattr(model, 'ssnc_mlp'):
         ssnc_mlp = model.ssnc_mlp
    elif isinstance(model, nn.Module) and hasattr(model, 'neuron_metrics'): # Direct MLP
         ssnc_mlp = model
    else:
         return # Can't visualize

    stats = ssnc_mlp.get_statistics()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f"SSNC Network Architecture"
    if epoch is not None:
        title += f" (Epoch {epoch})"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # === 1. Layer Structure Diagram ===
    ax1 = axes[0, 0]
    
    # Get layer sizes
    layer_sizes = [ssnc_mlp.input_dim] + ssnc_mlp.hidden_dims + [ssnc_mlp.output_dim]
    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(ssnc_mlp.hidden_dims))] + ['Output']
    
    # Draw layers as stacked bars
    max_size = max(layer_sizes)
    colors = ['lightblue'] + ['lightgreen'] * len(ssnc_mlp.hidden_dims) + ['lightsalmon']
    
    x_positions = range(len(layer_sizes))
    bars = ax1.bar(x_positions, layer_sizes, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add neuron counts on bars
    for i, (x, size) in enumerate(zip(x_positions, layer_sizes)):
        ax1.text(x, size + max_size * 0.02, str(size), ha='center', fontweight='bold')
    
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    ax1.set_ylabel('Number of Neurons')
    ax1.set_title('Layer Structure', fontweight='bold')
    ax1.set_ylim(0, max_size * 1.15)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # === 2. Neuron States per Layer ===
    ax2 = axes[0, 1]
    
    # Count neuron states per hidden layer
    layer_states = []
    for layer_idx in range(len(ssnc_mlp.hidden_dims)):
        active = 0
        marked = 0
        soft_deleted = 0
        
        # Check bounds before accessing ssnc_neurons
        if layer_idx >= len(ssnc_mlp.ssnc_neurons):
            continue
            
        for neuron_idx in range(len(ssnc_mlp.ssnc_neurons[layer_idx])):
            key = (layer_idx, neuron_idx)
            metrics = ssnc_mlp.neuron_metrics.get(key)
            if metrics:
                if metrics.hard_deleted:
                    continue
                elif metrics.soft_deleted:
                    soft_deleted += 1
                elif metrics.marked_for_deletion:
                    marked += 1
                else:
                    active += 1
        
        layer_states.append({'active': active, 'marked': marked, 'soft_deleted': soft_deleted})
    
    # Stacked bar chart
    x = np.arange(len(ssnc_mlp.hidden_dims))
    width = 0.6
    
    active_counts = [s['active'] for s in layer_states]
    marked_counts = [s['marked'] for s in layer_states]
    soft_del_counts = [s['soft_deleted'] for s in layer_states]
    
    ax2.bar(x, active_counts, width, label='Active', color='green', alpha=0.8)
    ax2.bar(x, marked_counts, width, bottom=active_counts, label='Marked', color='yellow', alpha=0.8)
    ax2.bar(x, soft_del_counts, width, 
            bottom=[a + m for a, m in zip(active_counts, marked_counts)],
            label='Soft-deleted', color='red', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Layer {i+1}' for i in range(len(ssnc_mlp.hidden_dims))])
    ax2.set_ylabel('Neuron Count')
    ax2.set_title('Neuron States by Layer', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # === 3. Weight Magnitude Distribution ===
    ax3 = axes[1, 0]
    
    all_weights = []
    layer_labels = []
    for i, layer in enumerate(ssnc_mlp.layers[:-1]):  # Exclude output layer
        weights = layer.weight.detach().cpu().numpy().flatten()
        all_weights.append(weights)
        layer_labels.append(f'Layer {i+1}')
    
    # Box plot of weight distributions
    bp = ax3.boxplot(all_weights, labels=layer_labels, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(all_weights)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('Weight Magnitude')
    ax3.set_title('Weight Distribution by Layer', fontweight='bold')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # === 4. Utility Distribution ===
    ax4 = axes[1, 1]
    
    layer_utilities = []
    for layer_idx in range(len(ssnc_mlp.hidden_dims)):
        utilities = []
        # Check bounds before accessing ssnc_neurons
        if layer_idx >= len(ssnc_mlp.ssnc_neurons):
            layer_utilities.append([0])
            continue
            
        for neuron_idx in range(len(ssnc_mlp.ssnc_neurons[layer_idx])):
            key = (layer_idx, neuron_idx)
            metrics = ssnc_mlp.neuron_metrics.get(key)
            if metrics and not metrics.hard_deleted:
                neuron = ssnc_mlp.ssnc_neurons[layer_idx][neuron_idx]
                utilities.append(neuron.utility)
        if utilities:
            layer_utilities.append(utilities)
        else:
            layer_utilities.append([0])
    
    # Violin plot for utility distribution
    parts = ax4.violinplot(layer_utilities, positions=range(len(layer_utilities)), 
                           showmeans=True, showmedians=True)
    
    # Color the violin plots
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_alpha(0.7)
    
    # Add threshold lines
    ax4.axhline(y=ssnc_mlp.deletion_threshold, color='red', linestyle='--', 
                alpha=0.7, label=f'Delete threshold ({ssnc_mlp.deletion_threshold})')
    ax4.axhline(y=ssnc_mlp.creation_threshold, color='green', linestyle='--', 
                alpha=0.7, label=f'Create threshold ({ssnc_mlp.creation_threshold})')
    
    ax4.set_xticks(range(len(layer_utilities)))
    ax4.set_xticklabels([f'Layer {i+1}' for i in range(len(layer_utilities))])
    ax4.set_ylabel('Utility Score')
    ax4.set_title('Utility Distribution by Layer', fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(-0.1, 1.0)
    
    plt.tight_layout()
    
    # Save
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'network_architecture.png', dpi=150, bbox_inches='tight')
