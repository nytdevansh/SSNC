import argparse
import sys
from pathlib import Path

# Add src to path to ensure imports work correctly
sys.path.append(str(Path(__file__).parent))

from src.trec6_model import CONFIG
from src.trainer import train

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSNC on TREC-6 Dataset')
    
    # Training Config
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'],
                        help=f"Number of epochs (default: {CONFIG['epochs']})")
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'],
                        help=f"Batch size (default: {CONFIG['batch_size']})")
    parser.add_argument('--lr', type=float, default=CONFIG['learning_rate'],
                        help=f"Learning rate (default: {CONFIG['learning_rate']})")
    parser.add_argument('--seed', type=int, default=CONFIG['seed'],
                        help=f"Random seed (default: {CONFIG['seed']})")
    
    # Model Config
    parser.add_argument('--glove', type=str, default=CONFIG.get('glove_path'),
                        help="Path to GloVe embeddings file")
    parser.add_argument('--encoder', type=str, choices=['glove', 'tfidf', 'word', 'simple'],
                        default=CONFIG.get('encoder_type', 'glove'),
                        help="Text encoder type to use")
    
    # SSNC Config
    parser.add_argument('--no_ssnc', action='store_true',
                        help="Disable SSNC (pure static training)")
    parser.add_argument('--burn_in', type=int, default=CONFIG['burn_in_epochs'],
                        help=f"Burn-in epochs before structural changes (default: {CONFIG['burn_in_epochs']})")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Update configuration with CLI arguments
    config = CONFIG.copy()
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['seed'] = args.seed
    config['encoder_type'] = args.encoder
    config['burn_in_epochs'] = args.burn_in
    
    if args.glove:
        config['glove_path'] = args.glove
        
    if args.no_ssnc:
        print("⚠️ Disabling SSNC features (Static Training Mode)")
        config['enable_soft_deletion'] = False
        config['enable_hard_deletion'] = False
        config['enable_creation'] = False
        
    # Run training
    train(config)

if __name__ == "__main__":
    main()
