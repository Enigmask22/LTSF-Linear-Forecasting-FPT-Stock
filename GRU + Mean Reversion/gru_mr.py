"""
FPT Stock Forecasting - V2.37 (Model Comparison with 50% Weight)
=================================================================

Goal: Prove that models can learn well WITHOUT heavy mean reversion dependency
Test: LSTM, GRU with 50% model weight (same as V2.32 BiLSTM baseline)

Usage: Change MODEL_TYPE to test different architectures
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import random
warnings.filterwarnings('ignore')

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_TYPE = "GRU"  # Options: "LSTM", "GRU", "GRU_Enhanced"
MODEL_WEIGHT = 0.5   # 50% model, 50% mean reversion - BEST so far (MSE=23.5)

print(f"\n{'='*60}")
print(f"Testing: {MODEL_TYPE} with {int(MODEL_WEIGHT*100)}% model weight")
print(f"{'='*60}")

# ==========================================
# DATASET
# ==========================================

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len=5):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# ==========================================
# FEATURES
# ==========================================

def create_features(df):
    df = df.copy().sort_values('time').reset_index(drop=True)
    
    for window in [5, 10, 20, 30, 50]:
        df[f'sma_{window}'] = df['close'].rolling(window, min_periods=1).mean()
        df[f'std_{window}'] = df['close'].rolling(window, min_periods=1).std()
        df[f'min_{window}'] = df['close'].rolling(window, min_periods=1).min()
        df[f'max_{window}'] = df['close'].rolling(window, min_periods=1).max()
    
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'ret_{lag}'] = df['close'].pct_change(lag)
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    df['bb_mid'] = df['close'].rolling(20, min_periods=1).mean()
    df['bb_std'] = df['close'].rolling(20, min_periods=1).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-10)
    
    df['price_position'] = (df['close'] - df['min_20']) / (df['max_20'] - df['min_20'] + 1e-10)
    df['volatility'] = df['ret_1'].rolling(20, min_periods=1).std()
    
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(10, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        df['volume_change'] = df['volume'].pct_change()
    
    df['time'] = pd.to_datetime(df['time'])
    df['dayofweek'] = df['time'].dt.dayofweek / 6.0
    df['day'] = df['time'].dt.day / 31.0
    df['month'] = df['time'].dt.month / 12.0
    
    return df

def prepare_data(df):
    df = create_features(df)
    feature_cols = ['close'] + [c for c in df.columns if c not in 
                               ['time', 'symbol', 'open', 'high', 'low', 'volume']]
    df = df[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
    return df.values, feature_cols

# ==========================================
# MODEL ARCHITECTURES
# ==========================================

class LSTMModel(nn.Module):
    """Standard LSTM - same architecture as common baselines"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2, pred_len=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pred_len)
        )
        
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(out[:, -1])


class GRUModel(nn.Module):
    """Standard GRU - lighter than LSTM, often performs similarly"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2, pred_len=5):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pred_len)
        )
        
    def forward(self, x):
        out, h = self.gru(x)
        return self.fc(out[:, -1])


def get_model(model_type, input_dim, pred_len=5):
    models = {
        "LSTM": LSTMModel(input_dim, hidden_dim=128, num_layers=3, pred_len=pred_len),
        "GRU": GRUModel(input_dim, hidden_dim=128, num_layers=3, pred_len=pred_len),
    }
    return models[model_type]

# ==========================================
# TRAINING
# ==========================================

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    print(f"\n[Training {MODEL_TYPE}]")
    print(f"  Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    print(f"\n  Epoch    Train Loss    Val Loss      Best Val    Patience")
    print(f"  " + "-"*55)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y[:, :, 0])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y[:, :, 0])
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  {epoch+1:5d}    {train_loss:10.6f}    {val_loss:8.6f}    {best_val_loss:8.6f}    {patience_counter:5d}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n  Best validation loss: {best_val_loss:.6f}")
    return model, best_val_loss

# ==========================================
# PREDICTION
# ==========================================

def generate_predictions(model, data_scaled, scaler, feature_cols, seq_len, n_steps):
    model.eval()
    predictions = []
    current_seq = data_scaled[-seq_len:].copy()
    close_idx = feature_cols.index('close')
    
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.FloatTensor(current_seq).unsqueeze(0).to(device)
            pred = model(x)
            next_close = pred[0, 0].cpu().numpy()
            
            next_features = current_seq[-1].copy()
            next_features[close_idx] = next_close
            current_seq = np.vstack([current_seq[1:], next_features])
            predictions.append(next_close)
    
    return np.array(predictions)

# ==========================================
# MEAN REVERSION
# ==========================================

def mean_reversion_forecast(last_price, target_mean, n_steps, speed=0.10):
    predictions = []
    current = last_price
    for _ in range(n_steps):
        current = current + speed * (target_mean - current)
        predictions.append(current)
    return np.array(predictions)

# ==========================================
# HYBRID COMBINATION (same as V2.32)
# ==========================================

def hybrid_forecast(model_preds, mr_preds, last_price, model_weight=0.5, transition_days=50):
    n_steps = len(model_preds)
    hybrid = np.zeros(n_steps)
    
    for i in range(n_steps):
        if i < 10:
            # Smooth start for continuity
            if i == 0:
                hybrid[i] = last_price + 0.3 * (mr_preds[i] - last_price)
            else:
                smooth_factor = (10 - i) / 10
                normal = model_weight * model_preds[i] + (1 - model_weight) * mr_preds[i]
                expected = hybrid[i-1] + (mr_preds[i] - mr_preds[i-1]) * 0.5
                hybrid[i] = smooth_factor * expected + (1 - smooth_factor) * normal
        else:
            # Transition: model weight decreases over time
            if i < transition_days:
                current_weight = model_weight * (1 - i / transition_days)
            else:
                current_weight = 0.1
            hybrid[i] = current_weight * model_preds[i] + (1 - current_weight) * mr_preds[i]
    
    return hybrid

# ==========================================
# MAIN
# ==========================================

def main():
    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv('FPT_train.csv')
    last_price = df['close'].iloc[-1]
    mean_365 = df['close'].iloc[-365:].mean()
    
    print(f"    Last price: {last_price:.2f}")
    print(f"    365-day mean (target): {mean_365:.2f}")
    
    # Prepare data
    print("\n[2] Preparing features...")
    data, feature_cols = prepare_data(df)
    print(f"    Features: {len(feature_cols)}")
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Split 80/20
    split = int(len(data_scaled) * 0.80)
    train_data = data_scaled[:split]
    val_data = data_scaled[split:]
    
    seq_len = 30
    pred_len = 5
    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_data, seq_len, pred_len)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"    Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Train model
    print(f"\n[3] Training {MODEL_TYPE}...")
    model = get_model(MODEL_TYPE, len(feature_cols), pred_len).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {params:,}")
    
    model, val_loss = train_model(model, train_loader, val_loader, epochs=100, patience=15)
    
    # Generate predictions
    print("\n[4] Generating predictions...")
    
    # Model predictions
    model_preds = generate_predictions(model, data_scaled, scaler, feature_cols, seq_len, 100)
    
    # Inverse transform
    close_idx = feature_cols.index('close')
    dummy = np.zeros((100, len(feature_cols)))
    dummy[:, close_idx] = model_preds
    model_preds_original = scaler.inverse_transform(dummy)[:, close_idx]
    
    print(f"\n    {MODEL_TYPE} Predictions:")
    print(f"      Day 1:   {model_preds_original[0]:.2f}")
    print(f"      Day 50:  {model_preds_original[49]:.2f}")
    print(f"      Day 100: {model_preds_original[-1]:.2f}")
    print(f"      Mean:    {model_preds_original.mean():.2f}")
    
    # Mean reversion
    mr_preds = mean_reversion_forecast(last_price, mean_365, 100, speed=0.10)
    print(f"\n    Mean Reversion:")
    print(f"      Day 1:   {mr_preds[0]:.2f}")
    print(f"      Day 50:  {mr_preds[49]:.2f}")
    print(f"      Day 100: {mr_preds[-1]:.2f}")
    print(f"      Mean:    {mr_preds.mean():.2f}")
    
    # Hybrid
    print(f"\n[5] Hybrid ({int(MODEL_WEIGHT*100)}% {MODEL_TYPE} + {int((1-MODEL_WEIGHT)*100)}% MR)...")
    hybrid = hybrid_forecast(model_preds_original, mr_preds, last_price, MODEL_WEIGHT)
    
    # Smoothing
    from scipy.ndimage import gaussian_filter1d
    final_preds = gaussian_filter1d(hybrid, sigma=1)
    
    print(f"\n    Final Predictions:")
    print(f"      Day 1:   {final_preds[0]:.2f} (gap: {abs(final_preds[0]-last_price):.2f})")
    print(f"      Day 50:  {final_preds[49]:.2f}")
    print(f"      Day 100: {final_preds[-1]:.2f}")
    print(f"      Mean:    {final_preds.mean():.2f}")
    print(f"      Target:  {mean_365:.2f}")
    print(f"      Gap to target: {abs(final_preds.mean() - mean_365):.2f}")
    
    # Save
    print("\n[6] Saving submission...")
    submission = pd.DataFrame({'id': range(1, 101), 'close': final_preds})
    submission.to_csv('submission.csv', index=False)
    
    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY: {MODEL_TYPE} with {int(MODEL_WEIGHT*100)}% model weight")
    print("="*60)
    print(f"Val Loss:     {val_loss:.6f}")
    print(f"Parameters:   {params:,}")
    print(f"Model Mean:   {model_preds_original.mean():.2f}")
    print(f"Final Mean:   {final_preds.mean():.2f}")
    print(f"Target Mean:  {mean_365:.2f}")
    print("="*60)
    print("\nSubmit to Kaggle to compare with:")
    print("  - BiLSTM 50% → MSE 27")
    print("  - TCN 20%    → MSE 24")
    print("  - TCN 50%    → MSE 46")
    print("="*60)
    
    return final_preds

if __name__ == "__main__":
    main()
