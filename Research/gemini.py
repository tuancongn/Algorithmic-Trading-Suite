"""
================================================================================
PROFESSIONAL AI TRADING STRATEGY - RECRUITMENT SUBMISSION
================================================================================

Ý TƯỞNG CHIẾN LƯỢC:
-------------------
Kết hợp Machine Learning Ensemble với Market Regime Awareness để:
1. Dự đoán xác suất giá tăng trong 24 periods (12 giờ) tới
2. Điều chỉnh position sizing theo volatility và market regime
3. Sử dụng trailing stop loss để bảo vệ lợi nhuận

QUY TRÌNH XỬ LÝ DỮ LIỆU:
------------------------
1. Data Cleaning: Resample về 30min để loại bỏ duplicate
2. Feature Engineering: Tạo 38 features từ Price/Volume/Time
   - Classical: MA, RSI, ATR, Lags, Distance to MA
   - Advanced: Efficiency Ratio, Shadow Asymmetry, Z-Score, etc.
3. Target Creation: Binary (1 nếu giá tăng >1.5% sau 12h)
4. Walk-Forward Training: Huấn luyện lại model mỗi 3 tháng

LOGIC CHIẾN LƯỢC:
-----------------
1. Entry: AI probability > threshold AND market regime cho phép
2. Position Size: Động dựa trên regime + volatility
3. Exit: Trailing SL 3% hoặc Take Profit 5%

METHODOLOGY:
-----------
- Train/Val/Test split để tránh selection bias
- Walk-Forward Optimization: Tune params trên validation, test trên unseen
- Ensemble 5 models: LightGBM, XGBoost, NN, RF, GradientBoosting

KẾT QUẢ:
--------
Xem phần báo cáo cuối chương trình.

================================================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import vectorbt as vbt
import json
import warnings
import os
import matplotlib.pyplot as plt
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from dateutil.relativedelta import relativedelta
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

# ==============================================================================
# 1. CONFIGURATION & UTILITIES
# ==============================================================================

def load_config(path='config_gemini.json'):
    """
    Load cấu hình từ file JSON.
    
    Returns:
        dict: Configuration dictionary chứa parameters cho feature engineering,
              training, models và optimization.
    """
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy file {path}")
        sys.exit(1)
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except:
        with open(path, 'r') as f:
            return json.load(f)


# ==============================================================================
# 2. MARKET REGIME DETECTION
# ==============================================================================

def detect_market_regime(df, lookback=100):
    """
    Phát hiện trạng thái thị trường dựa trên trend strength và momentum.
    
    Methodology:
    -----------
    1. Trend Strength: % khoảng cách giá so với SMA200
    2. Momentum: Rate of Change trong lookback periods
    3. Volatility: ATR normalized
    
    Classification Rules:
    --------------------
    - STRONG_BULL: Momentum > 50% và Trend > 20% (Bull market mạnh)
    - BULL: Momentum > 15% và Trend > 5% (Uptrend vừa phải)
    - SIDEWAYS: Không rõ xu hướng (Default state)
    - BEAR: Momentum < -10% và Trend < -5% (Downtrend)
    - STRONG_BEAR: Momentum < -30% và Trend < -15% (Bear market mạnh)
    
    Args:
        df: DataFrame với OHLCV data
        lookback: Periods để tính momentum (default: 100 = 50 giờ)
    
    Returns:
        pd.Series: Market regime cho mỗi timestamp
    """
    regimes = pd.Series(index=df.index, dtype=str)
    
    # Tính các chỉ báo
    sma_50 = df['close'].rolling(50).mean()
    sma_200 = df['close'].rolling(200).mean()
    
    # Trend Strength: % giá trên SMA200
    trend = (df['close'] - sma_200) / sma_200
    
    # Volatility: ATR normalized
    atr = vbt.ATR.run(df['high'], df['low'], df['close'], window=14).atr
    vol_norm = atr / df['close']
    
    # Momentum: ROC trong lookback periods
    momentum = df['close'].pct_change(lookback)
    
    for i in range(len(df)):
        if i < 200:  # Cần đủ dữ liệu cho SMA200
            regimes.iloc[i] = 'SIDEWAYS'
            continue
            
        t = trend.iloc[i]
        m = momentum.iloc[i]
        v = vol_norm.iloc[i]
        
        # Classification logic
        if m > 0.5 and t > 0.2:
            regimes.iloc[i] = 'STRONG_BULL'
        elif m > 0.15 and t > 0.05:
            regimes.iloc[i] = 'BULL'
        elif m < -0.3 and t < -0.15:
            regimes.iloc[i] = 'STRONG_BEAR'
        elif m < -0.1 and t < -0.05:
            regimes.iloc[i] = 'BEAR'
        else:
            regimes.iloc[i] = 'SIDEWAYS'
    
    return regimes


def calculate_position_size(regime, base_size=1.0, volatility=None):
    """
    Tính position size động dựa trên market regime và volatility.
    
    Risk Management Logic:
    ---------------------
    - STRONG_BULL: 100% size - Tận dụng bull run
    - BULL: 80% size - Uptrend nhưng thận trọng
    - SIDEWAYS: 50% size - Thị trường không rõ
    - BEAR: 30% size - Defensive
    - STRONG_BEAR: 0% size - Tránh hoàn toàn
    
    Volatility Adjustment:
    ---------------------
    - ATR > 5%: Giảm size xuống 70% (high risk)
    - ATR > 3%: Giảm size xuống 85% (medium risk)
    
    Args:
        regime: Market regime string
        base_size: Base position size (default: 1.0 = 100%)
        volatility: ATR normalized (optional)
    
    Returns:
        float: Adjusted position size [0, 1]
    """
    regime_multipliers = {
        'STRONG_BULL': 1.0,
        'BULL': 0.8,
        'SIDEWAYS': 0.5,
        'BEAR': 0.3,
        'STRONG_BEAR': 0.0
    }
    
    size = base_size * regime_multipliers.get(regime, 0.5)
    
    # Volatility adjustment
    if volatility is not None:
        if volatility > 0.05:
            size *= 0.7
        elif volatility > 0.03:
            size *= 0.85
    
    return max(size, 0.0)


def get_adaptive_threshold(regime, base_threshold=0.56):
    """
    Điều chỉnh AI confidence threshold dựa trên market regime.
    
    Rationale:
    ---------
    - Bull market: Giảm threshold → Trade tích cực hơn để catch momentum
    - Bear market: Tăng threshold → Chọn lọc kỹ để tránh false signals
    
    Args:
        regime: Market regime string
        base_threshold: Base AI threshold (default: 0.56)
    
    Returns:
        float: Adjusted threshold
    """
    adjustments = {
        'STRONG_BULL': -0.08,  # 0.56 → 0.48
        'BULL': -0.04,         # 0.56 → 0.52
        'SIDEWAYS': 0.0,
        'BEAR': +0.05,         # 0.56 → 0.61
        'STRONG_BEAR': +0.15   # 0.56 → 0.71
    }
    
    return base_threshold + adjustments.get(regime, 0.0)


# ==============================================================================
# 3. ADVANCED FEATURE ENGINEERING
# ==============================================================================

def add_advanced_features(df, config):
    """
    Tạo 8 đặc trưng nâng cao dựa trên nghiên cứu định lượng hiện đại.
    
    8 Super Features:
    ----------------
    1. Volatility-Scaled Returns: Log returns chuẩn hóa theo volatility
       → Giúp so sánh biến động trong các giai đoạn khác nhau
    
    2. Efficiency Ratio (Kaufman): Đo độ "mượt" của xu hướng
       → 1.0 = xu hướng hoàn hảo, 0.0 = nhiễu hoàn toàn
    
    3. Relative Volume Intensity: Volume so với MA
       → Phát hiện breakout/breakdown có volume support
    
    4. Shadow Asymmetry: Tỷ lệ bóng trên/dưới nến
       → Đo áp lực mua/bán trong phiên
    
    5. Rolling Z-Score: Khoảng cách chuẩn hóa so với MA
       → Mean reversion signal
    
    6. Volatility Regime Ratio: Range vs Body
       → Phân biệt Doji (indecision) vs Marubozu (conviction)
    
    7. Cyclical Time Encoding: Sin/Cos của giờ
       → Capture intraday patterns (23h gần 0h)
    
    8. Trend-Momentum Interaction: MACD × RSI
       → Kết hợp trend strength và momentum
    
    Args:
        df: DataFrame với OHLCV data
        config: Configuration dictionary
    
    Returns:
        DataFrame: df với 8 features mới
    """
    fp = config['feature_params']
    
    # 1. Volatility-Scaled Log Returns
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['feat_vol_scaled_ret'] = log_ret / log_ret.rolling(window=fp['volatility_window']).std()
    
    # 2. Efficiency Ratio
    change = (df['close'] - df['close'].shift(fp['efficiency_window'])).abs()
    volatility = (df['close'] - df['close'].shift(1)).abs().rolling(window=fp['efficiency_window']).sum()
    df['feat_efficiency_ratio'] = change / volatility.replace(0, 1)
    
    # 3. Relative Volume Intensity
    df['feat_rel_vol'] = df['volume'] / df['volume'].rolling(window=fp['volume_window']).mean()
    
    # 4. Shadow Asymmetry
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    candle_range = df['high'] - df['low']
    df['feat_shadow_asym'] = (upper_shadow - lower_shadow) / candle_range.replace(0, 1)
    
    # 5. Rolling Z-Score
    ma = df['close'].rolling(window=fp['zscore_window']).mean()
    std = df['close'].rolling(window=fp['zscore_window']).std()
    df['feat_z_score'] = (df['close'] - ma) / std
    
    # 6. Volatility Regime Ratio
    range_vol = df['high'] - df['low']
    close_vol = (df['close'] - df['open']).abs()
    df['feat_vol_regime'] = range_vol / close_vol.replace(0, 0.0001)
    
    # 7. Cyclical Time Encoding
    df['feat_hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['feat_hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # 8. Trend-Momentum Interaction
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    trend_width = ema_fast - ema_slow
    momentum_state = vbt.RSI.run(df['close']).rsi - 50
    df['feat_trend_mom_interaction'] = trend_width * momentum_state
    
    return df


def prepare_data(config):
    """
    Load và xử lý dữ liệu ETHUSDT.
    
    Processing Pipeline:
    -------------------
    1. Load CSV và validate columns
    2. Convert timestamp và set as index
    3. Resample về 30min để remove duplicates
    4. Feature Engineering:
       - Classical features (MA, RSI, ATR, lags)
       - Advanced features (8 super features)
    5. Target Creation: Binary label (price up >1.5% in 12h)
    6. Clean NaN và infinite values
    
    Returns:
        DataFrame: Processed data với features và target
    """
    print("⏳ Đang tải và xử lý dữ liệu (Advanced Preprocessing)...")
    
    try:
        df = pd.read_csv(config['file_path'])
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file CSV dữ liệu.")
        sys.exit(1)

    df.columns = [c.lower().strip() for c in df.columns]
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.set_index('timestamp').sort_index()
    else:
        print("⚠️ Cảnh báo: Không tìm thấy cột 'timestamp'.")
        sys.exit(1)

    cols_to_numeric = ['open', 'high', 'low', 'close', 'volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()

    print("   -> Đang Resample về khung 30 phút để loại bỏ trùng lặp...")
    df_resampled = df.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    df = df_resampled.dropna(subset=['close'])
    print(f"   -> Dữ liệu sạch: {len(df)} nến 30-phút.")

    # Classical Feature Engineering
    df['ret'] = df['close'].pct_change()
    
    for lag in config['feature_params']['lags']:
        df[f'ret_lag_{lag}'] = df['ret'].shift(lag)
        df[f'vol_lag_{lag}'] = df['volume'].shift(lag)

    for window in config['feature_params']['ma_windows']:
        df[f'ma_{window}'] = vbt.MA.run(df['close'], window).ma
        df[f'dist_ma_{window}'] = (df['close'] / df[f'ma_{window}']) - 1

    rsi = vbt.RSI.run(df['close'], window=config['feature_params']['rsi_window']).rsi
    df['rsi'] = rsi

    atr = vbt.ATR.run(df['high'], df['low'], df['close'], window=config['feature_params']['atr_window']).atr
    df['atr_rel'] = atr / df['close']

    # Advanced Feature Engineering
    print("   -> Đang tính toán 8 Siêu Đặc Trưng (Smart Features)...")
    df = add_advanced_features(df, config)

    # Target Creation
    k = config['training_params']['prediction_horizon_k']
    thresh = config['training_params']['target_threshold']
    future_ret = df['close'].shift(-k) / df['close'] - 1
    df['target'] = (future_ret > thresh).astype(int)

    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"✅ Dữ liệu sẵn sàng: {df.shape[0]} dòng, {df.shape[1]} đặc trưng.")
    return df


# ==============================================================================
# 4. WALK-FORWARD TRAINING
# ==============================================================================

def train_walk_forward(df, config):
    """
    Walk-Forward Training để tránh look-ahead bias.
    
    Methodology:
    -----------
    1. Bắt đầu với initial_train_months (24 tháng) để train
    2. Mỗi update_months (3 tháng), retrain model với data mới
    3. Ensemble 5 models để giảm overfitting:
       - LightGBM: Gradient boosting nhanh
       - XGBoost: Robust gradient boosting
       - Neural Network: Non-linear patterns
       - Random Forest: Bagging ensemble
       - Gradient Boosting: Classic boosting
    
    Why Ensemble?
    ------------
    - Giảm variance (RF giỏi handling noise)
    - Giảm bias (GBM giỏi capturing complex patterns)
    - More stable predictions
    
    Returns:
        pd.Series: AI probability predictions cho toàn bộ timeline
    """
    non_feature_cols = ['target', 'open', 'high', 'low', 'close', 'volume', 'ret']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    initial_train_months = config['training_params']['initial_train_months']
    update_months = config['training_params']['update_months']
    
    start_date = df.index[0]
    end_date = df.index[-1]
    current_date = start_date + relativedelta(months=initial_train_months)
    
    predictions = []
    indices = []
    
    params_lgb = config['models']['lightgbm']
    params_xgb = config['models']['xgboost']
    params_nn = config['models']['neural_network']
    params_rf = config['models'].get('random_forest', {'n_estimators': 100, 'max_depth': 10})
    params_gb = config['models'].get('gradient_boosting', {'n_estimators': 100, 'learning_rate': 0.05})
    
    print(f"🚀 Bắt đầu Walk-Forward Training (5 Models) từ {current_date.date()}...")
    
    while current_date < end_date:
        next_date = current_date + relativedelta(months=update_months)
        if next_date > end_date: next_date = end_date
            
        X_train = X.loc[:current_date]
        y_train = y.loc[:current_date]
        X_test = X.loc[current_date:next_date]
        
        if not X_test.empty and not X_train.empty and X_test.index[0] == X_train.index[-1]:
             X_test = X_test.iloc[1:]

        if len(X_test) == 0: 
            current_date = next_date
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train 5 models
        model_lgb = lgb.LGBMClassifier(**params_lgb)
        model_lgb.fit(X_train, y_train)
        pred_lgb = model_lgb.predict_proba(X_test)[:, 1]
        
        model_xgb = xgb.XGBClassifier(**params_xgb)
        model_xgb.fit(X_train, y_train)
        pred_xgb = model_xgb.predict_proba(X_test)[:, 1]
        
        model_nn = MLPClassifier(**params_nn)
        model_nn.fit(X_train_scaled, y_train)
        pred_nn = model_nn.predict_proba(X_test_scaled)[:, 1]

        model_rf = RandomForestClassifier(**params_rf)
        model_rf.fit(X_train, y_train)
        pred_rf = model_rf.predict_proba(X_test)[:, 1]

        model_gb = GradientBoostingClassifier(**params_gb)
        model_gb.fit(X_train, y_train)
        pred_gb = model_gb.predict_proba(X_test)[:, 1]
        
        # Equal-weight ensemble
        ensemble_pred = (pred_lgb + pred_xgb + pred_nn + pred_rf + pred_gb) / 5
        
        predictions.extend(ensemble_pred)
        indices.extend(X_test.index)
        
        current_date = next_date

    pred_series = pd.Series(predictions, index=indices, name='signal_prob')
    pred_series = pred_series[~pred_series.index.duplicated(keep='last')]
    
    full_signal = df.join(pred_series, how='left')['signal_prob']
    
    print("✅ Hoàn tất huấn luyện Walk-Forward.")
    return full_signal.fillna(0)


# ==============================================================================
# 5. PROPER VALIDATION & BACKTESTING (FIX SELECTION BIAS)
# ==============================================================================

def split_data(df, train_ratio=0.5, val_ratio=0.25):
    """
    Chia data thành Train/Validation/Test để tránh selection bias.
    
    Split Strategy:
    --------------
    - Train: 50% đầu - Để train models (Walk-forward)
    - Validation: 25% giữa - Để tune hyperparameters
    - Test: 25% cuối - Để đánh giá cuối cùng (UNSEEN)
    
    Critical:
    --------
    Hyperparameters CHỈ được tune trên Validation set.
    Test set KHÔNG BAO GIỜ được nhìn thấy cho đến evaluation cuối.
    
    Args:
        df: Full DataFrame
        train_ratio: Tỷ lệ training data
        val_ratio: Tỷ lệ validation data
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"\n📊 DATA SPLIT:")
    print(f"   Train: {train_df.index[0].date()} → {train_df.index[-1].date()} ({len(train_df)} samples)")
    print(f"   Val:   {val_df.index[0].date()} → {val_df.index[-1].date()} ({len(val_df)} samples)")
    print(f"   Test:  {test_df.index[0].date()} → {test_df.index[-1].date()} ({len(test_df)} samples)")
    
    return train_df, val_df, test_df


def optimize_on_validation(val_df, signal_probs, regimes, config):
    """
    Tune hyperparameters trên VALIDATION SET (KHÔNG phải test set).
    
    Parameters to Optimize:
    ----------------------
    - AI threshold: [0.53, 0.56, 0.59]
    - Stop loss: [2%, 3%, 4%]
    - Take profit: [5%, 6%, 7%]
    - Mode: [static, adaptive]
    
    Objective:
    ---------
    Maximize Sharpe Ratio trên validation set
    
    Returns:
        dict: Best parameters found on validation
    """
    print("\n🔍 HYPERPARAMETER OPTIMIZATION (Validation Set Only)...")
    
    price = val_df['close']
    signal_probs_val = signal_probs.loc[val_df.index]
    regimes_val = regimes.loc[val_df.index]
    
    atr = vbt.ATR.run(val_df['high'], val_df['low'], val_df['close'], window=14).atr
    atr_pct = atr / val_df['close']
    
    thresholds = config['optimization_params']['threshold_range']
    sl_stops = config['optimization_params']['sl_range']
    tp_stops = config['optimization_params'].get('tp_range', [None])
    modes = ['static', 'adaptive']
    
    best_sharpe = -999
    best_params = {}
    
    count = 0
    total = len(thresholds) * len(sl_stops) * len(tp_stops) * len(modes)
    
    for mode in modes:
        for base_thresh in thresholds:
            for sl in sl_stops:
                for tp in tp_stops:
                    
                    if mode == 'static':
                        entries = (signal_probs_val > base_thresh).shift(1).fillna(False)
                        size = 1.0
                    else:  # adaptive
                        entries_list = []
                        sizes_list = []
                        
                        for i in range(len(val_df)):
                            regime = regimes_val.iloc[i]
                            vol = atr_pct.iloc[i]
                            
                            thresh = get_adaptive_threshold(regime, base_thresh)
                            entry = signal_probs_val.iloc[i] > thresh
                            pos_size = calculate_position_size(regime, 1.0, vol)
                            
                            entries_list.append(entry)
                            sizes_list.append(pos_size)
                        
                        entries = pd.Series(entries_list, index=val_df.index).shift(1).fillna(False)
                        size = pd.Series(sizes_list, index=val_df.index)
                    
                    pf = vbt.Portfolio.from_signals(
                        price,
                        entries=entries,
                        sl_stop=sl,
                        tp_stop=tp,
                        size=size,
                        freq='30min',
                        init_cash=config['optimization_params']['initial_capital'],
                        fees=config['optimization_params']['fees'],
                        sl_trail=True,
                        size_type='percent',
                        accumulate=False
                    )
                    
                    if pf.trades.count() > 5:
                        sharpe = pf.sharpe_ratio()
                    else:
                        sharpe = -1.0

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {
                            'Mode': mode,
                            'Threshold': base_thresh,
                            'StopLoss': sl,
                            'TakeProfit': tp
                        }
                    
                    count += 1
                    if count % 10 == 0:
                        print(f"   ... Tested {count}/{total} combinations (Best Sharpe: {best_sharpe:.2f})")
    
    print(f"\n✅ Best Params on Validation: {best_params} (Sharpe: {best_sharpe:.2f})")
    return best_params


def backtest_on_test(test_df, signal_probs, regimes, best_params, config):
    """
    Backtest với best parameters trên TEST SET (unseen data).
    
    Critical:
    --------
    Đây là lần ĐẦU TIÊN test set được sử dụng.
    Không có bất kỳ tuning nào trên test set.
    
    Returns:
        Portfolio: VectorBT portfolio object
    """
    print("\n🎯 FINAL BACKTEST (Test Set - Unseen Data)...")
    
    price = test_df['close']
    signal_probs_test = signal_probs.loc[test_df.index]
    regimes_test = regimes.loc[test_df.index]
    
    atr = vbt.ATR.run(test_df['high'], test_df['low'], test_df['close'], window=14).atr
    atr_pct = atr / test_df['close']
    
    mode = best_params['Mode']
    base_thresh = best_params['Threshold']
    sl = best_params['StopLoss']
    tp = best_params['TakeProfit']
    
    if mode == 'static':
        entries = (signal_probs_test > base_thresh).shift(1).fillna(False)
        size = 1.0
    else:
        entries_list = []
        sizes_list = []
        
        for i in range(len(test_df)):
            regime = regimes_test.iloc[i]
            vol = atr_pct.iloc[i]
            
            thresh = get_adaptive_threshold(regime, base_thresh)
            entry = signal_probs_test.iloc[i] > thresh
            pos_size = calculate_position_size(regime, 1.0, vol)
            
            entries_list.append(entry)
            sizes_list.append(pos_size)
        
        entries = pd.Series(entries_list, index=test_df.index).shift(1).fillna(False)
        size = pd.Series(sizes_list, index=test_df.index)
    
    pf = vbt.Portfolio.from_signals(
        price,
        entries=entries,
        sl_stop=sl,
        tp_stop=tp,
        size=size,
        freq='30min',
        init_cash=config['optimization_params']['initial_capital'],
        fees=config['optimization_params']['fees'],
        sl_trail=True,
        size_type='percent',
        accumulate=False
    )
    
    return pf


# ==============================================================================
# 6. FEATURE IMPORTANCE & EXPLAINABILITY
# ==============================================================================

def analyze_feature_importance(df, config):
    """
    Phân tích feature importance để hiểu model quyết định dựa trên gì.
    
    Method:
    ------
    Train LightGBM model và extract feature importance.
    Visualize top 20 features.
    
    Returns:
        DataFrame: Feature importance scores
    """
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS...")
    
    non_feature_cols = ['target', 'open', 'high', 'low', 'close', 'volume', 'ret']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    X = df[feature_cols]
    y = df['target']
    
    # Train model
    model = lgb.LGBMClassifier(**config['models']['lightgbm'])
    model.fit(X, y)
    
    # Extract importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    top_20 = importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['Importance'])
    plt.yticks(range(len(top_20)), top_20['Feature'])
    plt.xlabel('Importance Score')
    plt.title('Top 20 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    
    print("   ✅ Đã lưu biểu đồ: feature_importance.png")
    return importance_df


# ==============================================================================
# 7. TRADE ANALYSIS & REPORTING
# ==============================================================================

def analyze_trades(portfolio):
    """
    Phân tích chi tiết các giao dịch: Winning vs Losing trades.
    
    Insights:
    --------
    - Thời gian giữ lệnh trung bình
    - P&L distribution
    - Win/Loss patterns
    
    Returns:
        dict: Trade statistics
    """
    trades = portfolio.trades.records
    if len(trades) == 0:
        return {}
    
    df = pd.DataFrame(trades)
    
    # Split winning vs losing
    winning = df[df['pnl'] > 0]
    losing = df[df['pnl'] <= 0]
    
    stats = {
        'Total Trades': len(df),
        'Winning Trades': len(winning),
        'Losing Trades': len(losing),
        'Win Rate': len(winning) / len(df) if len(df) > 0 else 0,
        'Avg Win': winning['return'].mean() if len(winning) > 0 else 0,
        'Avg Loss': losing['return'].mean() if len(losing) > 0 else 0,
        'Avg Win/Loss Ratio': abs(winning['return'].mean() / losing['return'].mean()) if len(losing) > 0 and len(winning) > 0 else 0,
        'Best Trade': df['return'].max(),
        'Worst Trade': df['return'].min()
    }
    
    # Visualize P&L distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['return'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='Breakeven')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.title('Trade Return Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    cumulative_pnl = df['pnl'].cumsum()
    plt.plot(cumulative_pnl, color='green', linewidth=2)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative P&L ($)')
    plt.title('Cumulative P&L Over Trades')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trade_analysis.png', dpi=150)
    plt.close()
    
    print("\n📊 TRADE ANALYSIS:")
    for key, val in stats.items():
        if 'Rate' in key or 'Ratio' in key or 'Win' in key or 'Loss' in key or 'Trade' in key:
            if isinstance(val, float) and 'Rate' in key:
                print(f"   {key:25s}: {val:.2%}")
            elif isinstance(val, float):
                print(f"   {key:25s}: {val:.2f}")
            else:
                print(f"   {key:25s}: {val}")
    
    print("   ✅ Đã lưu biểu đồ: trade_analysis.png")
    return stats


def export_trade_log(portfolio, output_file='trade_log.xlsx'):
    """Export chi tiết từng giao dịch ra Excel."""
    raw_records = portfolio.trades.records
    if len(raw_records) == 0: 
        return

    log_df = pd.DataFrame(raw_records)
    
    try:
        time_index = portfolio.wrapper.index
        log_df['Entry Date'] = time_index[log_df['entry_idx']].strftime('%d/%m/%Y %H:%M:%S')
        log_df['Exit Date'] = time_index[log_df['exit_idx']].strftime('%d/%m/%Y %H:%M:%S')
    except:
        log_df['Entry Date'] = log_df['entry_idx']
        log_df['Exit Date'] = log_df['exit_idx']

    rename_map = {
        'entry_price': 'Entry Price ($)',
        'exit_price': 'Exit Price ($)',
        'pnl': 'PnL ($)',
        'return': 'Return (%)',
        'direction': 'Direction',
        'status': 'Status'
    }
    log_df = log_df.rename(columns=rename_map)
    
    init_cash = portfolio.init_cash
    log_df['Cumulative PnL'] = log_df['PnL ($)'].cumsum()
    log_df['Equity After Trade'] = init_cash + log_df['Cumulative PnL']
    
    cols_order = ['Entry Date', 'Entry Price ($)', 'Exit Date', 'Exit Price ($)', 'Direction', 'PnL ($)', 'Return (%)', 'Equity After Trade', 'Status']
    final_cols = [c for c in cols_order if c in log_df.columns]
    
    try:
        log_df[final_cols].to_excel(output_file, index=False)
        print(f"\n📄 Đã xuất nhật ký giao dịch: {output_file}")
    except:
        pass


# ==============================================================================
# 8. COMPREHENSIVE REPORTING
# ==============================================================================

def generate_report(test_pf, bh_pf, params, test_df, regimes):
    """Báo cáo cuối cùng"""
    print("\n" + "="*80)
    print("🏆 BÁO CÁO KẾT QUẢ CUỐI CÙNG")
    print("="*80)
    
    print(f"\n1. CẤU HÌNH (Tuned on Validation):")
    print(f"   Mode        : {params['Mode'].upper()}")
    print(f"   Threshold   : {params['Threshold']}")
    print(f"   Stop Loss   : {params['StopLoss']*100:.1f}%")
    print(f"   Take Profit : {params['TakeProfit']*100:.1f}%" if params['TakeProfit'] else "   Take Profit : None")
    print(f"   Test Period : {test_df.index[0].date()} → {test_df.index[-1].date()}")
    
    reg_dist = regimes.loc[test_df.index].value_counts()
    print(f"\n2. MARKET REGIME (Test Set):")
    for r, c in reg_dist.items():
        print(f"   {r:12s}: {c/len(test_df)*100:5.1f}%")
    
    ai = {
        'Return': test_pf.total_return(),
        'Annual': test_pf.annualized_return(),
        'Sharpe': test_pf.sharpe_ratio(),
        'Sortino': test_pf.sortino_ratio(),
        'MaxDD': test_pf.max_drawdown(),
        'Trades': test_pf.trades.count(),
        'WinRate': test_pf.trades.win_rate(),
        'PF': test_pf.trades.profit_factor()
    }
    
    bh = {
        'Return': bh_pf.total_return(),
        'Annual': bh_pf.annualized_return(),
        'Sharpe': bh_pf.sharpe_ratio(),
        'MaxDD': bh_pf.max_drawdown()
    }
    
    print(f"\n3. HIỆU SUẤT (Test Set):")
    print("-" * 70)
    print(f"{'Metric':<20} | {'AI':<15} | {'Buy&Hold':<15} | {'Result'}")
    print("-" * 70)
    
    def cmp(name, v1, v2, pct=False):
        fmt = "{:.2%}" if pct else "{:.2f}"
        better = "✅ AI" if v1 > v2 else "❌ B&H" if name != 'MaxDD' else ("✅ AI" if v1 > v2 else "❌ B&H")
        print(f"{name:<20} | {fmt.format(v1):<15} | {fmt.format(v2):<15} | {better}")
    
    cmp('Total Return', ai['Return'], bh['Return'], True)
    cmp('Annual Return', ai['Annual'], bh['Annual'], True)
    cmp('Sharpe Ratio', ai['Sharpe'], bh['Sharpe'])
    cmp('Max Drawdown', ai['MaxDD'], bh['MaxDD'], True)
    print("-" * 70)
    
    print(f"\n4. TRADE STATS:")
    print(f"   Trades      : {ai['Trades']}")
    print(f"   Win Rate    : {ai['WinRate']:.2%}")
    print(f"   Profit Fctr : {ai['PF']:.2f}")
    
    print("\n" + "="*80)
    pass_req = ai['Sharpe'] > 0.3 and ai['Annual'] > 0.15
    print(f"{'✅ ĐẠT YÊU CẦU' if pass_req else '⚠️ CẦN CẢI THIỆN'}")
    
    # Chart
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(test_pf.value(), label='AI Strategy', linewidth=2)
        plt.plot(bh_pf.value(), label='Buy & Hold', linestyle='--', alpha=0.7)
        plt.title('AI vs Buy & Hold (Test Set)')
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('result.png', dpi=150)
        print("📊 Đã lưu: result.png")
    except:
        pass

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    config = load_config()
    df = prepare_data(config)
    
    # Detect regimes for full data
    print("\n🔍 Detecting Market Regimes...")
    regimes = detect_market_regime(df)
    
    # Train/Val/Test split
    train_df, val_df, test_df = split_data(df)
    
    # Walk-forward training (on train set only for prediction generation)
    signals = train_walk_forward(df, config)
    
    # Optimize on Validation
    best_params = optimize_on_validation(val_df, signals, regimes, config)
    
    # Test on unseen data
    test_pf = backtest_on_test(test_df, signals, regimes, best_params, config)
    
    # Buy & Hold baseline
    bh_pf = vbt.Portfolio.from_holding(
        test_df['close'],
        init_cash=config['optimization_params']['initial_capital'],
        fees=config['optimization_params']['fees'],
        freq='30min'
    )
    
    # Report
    generate_report(test_pf, bh_pf, best_params, test_df, regimes)
    export_trade_log(test_pf)

if __name__ == "__main__":
    main()

