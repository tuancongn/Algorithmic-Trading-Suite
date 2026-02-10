import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
import vectorbt as vbt
from colorama import Fore, Style, init
import warnings
import itertools
import traceback

warnings.filterwarnings("ignore")
init(autoreset=True)

SYMBOL = "BTCUSDT"
TIMEFRAME = "1D"

BYBIT_INTERVAL_MAP = {
    "1min": "1", "5min": "5", "15min": "15", "30min": "30",
    "1h": "60", "4h": "240", "12h": "720", 
    "1D": "D", "3D": "3D", "1W": "W", "30D": "M"
}
BYBIT_INTERVAL = BYBIT_INTERVAL_MAP.get(TIMEFRAME) 

DAYS_BACK = 1500

CASH = 100_000_000
LEVERAGE = 50 
COMMISSION = 0.000000
MARGIN_PER_TRADE = 0.3
OFFSET_PERCENT = 0.0005

def fetch_historical_data(symbol, interval, days):
    print(f"{Fore.CYAN}--- ĐANG TẢI DỮ LIỆU {symbol} ({days} NGÀY QUA) ---")
    session = HTTP(testnet=False)
    limit = 1000
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_kline = []
    
    while True:
        try:
            resp = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit, end=end_time)
            data = resp['result']['list']
            if not data: break
            all_kline.extend(data)
            last_timestamp = int(data[-1][0])
            end_time = last_timestamp - 1
            print(f"Đã tải đến: {datetime.fromtimestamp(last_timestamp/1000)}...", end='\r')
            if last_timestamp <= start_time: break
            time.sleep(0.05)
        except Exception as e:
            print(f"{Fore.RED}Lỗi tải data: {e}")
            break
            
    df = pd.DataFrame(all_kline, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover'])

    if df.empty:
        print(f"\n{Fore.RED}❌ LỖI: Không tải được dữ liệu nào! Kiểm tra lại Symbol hoặc Interval ('{interval}').")
        return df
    
    df = df.iloc[::-1].reset_index(drop=True)
    df['Time'] = pd.to_datetime(df['Time'].astype(int), unit='ms')
    df.set_index('Time', inplace=True)
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        df[c] = df[c].astype(float)
        
    df = df[df.index >= datetime.fromtimestamp(start_time/1000)]
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"\n{Fore.GREEN}✅ Tải hoàn tất! Tổng cộng: {len(df)} nến.")
    return df

def calculate_indicators(df, params=None):
    """Tính toán tất cả chỉ báo cần thiết"""
    if params is None:
        params = {}
    
    ema_period = params.get('ema_period', 200)
    rsi_period = params.get('rsi_period', 18)
    adx_threshold = params.get('adx_threshold', 20)
    atr_multiplier = params.get('atr_multiplier', 3.0)

    close_price = df['Close'].astype(np.float64)
    high_price = df['High'].astype(np.float64)
    low_price = df['Low'].astype(np.float64)
    
    # Các chỉ báo cơ bản
    df['ema'] = vbt.talib('EMA').run(df['Close'], timeperiod=ema_period).real
    df['rsi'] = vbt.talib('RSI').run(df['Close'], timeperiod=rsi_period).real
    
    # ATR
    df['atr'] = vbt.talib('ATR').run(
        df['High'], df['Low'], df['Close'], timeperiod=14
    ).real
    
    # ADX
    df['adx'] = vbt.talib('ADX').run(
        df['High'], df['Low'], df['Close'], timeperiod=14
    ).real
    
    # Volume MA
    df['vol_ma'] = df['Volume'].rolling(20).mean()
    
    # MR indicators
    df['mr_sma'] = df['Close'].rolling(20).mean()
    df['mr_std'] = df['Close'].rolling(20).std()
    
    # Bands for MR
    df['mr_upper_band'] = df['mr_sma'] + (2.0 * df['mr_std'])
    df['mr_lower_band'] = df['mr_sma'] - (2.0 * df['mr_std'])
    
    # Risk distance
    df['dist'] = df['atr'] * atr_multiplier
    
    # Risk per trade based on regime
    df['risk_pct'] = np.where(df['adx'] < adx_threshold, 0.001, 0.01)
    
    return df

def generate_signals(df, params):
    """Tối ưu tín hiệu: Gồng lãi Long, hạn chế Short ẩu"""
    
    ema_period = params.get('ema_period', 200)
    rsi_period = params.get('rsi_period', 18)
    adx_threshold = params.get('adx_threshold', 20)
    
    # Điều kiện môi trường
    trend_strong = df['adx'] >= adx_threshold
    vol_confirm = df['Volume'] >= df['vol_ma']
    
    # --- ENTRY ---
    # Long Trend: Giá > EMA, EMA dốc lên, RSI chưa quá nóng (<75)
    long_condition = (
        (df['Close'] > df['ema']) &
        (df['ema'] > df['ema'].shift(1)) &
        (df['rsi'] < 75) & 
        trend_strong &
        vol_confirm
    )
    
    # Short Trend: Chỉ Short khi gãy EMA và RSI hồi lên nhưng yếu
    # Thêm bộ lọc: Giá phải nằm dưới EMA 200 dài hạn (nếu ema_period nhỏ)
    # Ở đây ta dùng chính ema hiện tại nhưng yêu cầu RSI cao hơn chút để tránh bán đáy
    short_condition = (
        (df['Close'] < df['ema']) &
        (df['ema'] < df['ema'].shift(1)) &
        (df['rsi'] > 45) & (df['rsi'] < 60) & 
        trend_strong &
        vol_confirm
    )
    
    # Entries logic
    entries_long = long_condition & ~long_condition.shift(1).fillna(False)
    entries_short = short_condition & ~short_condition.shift(1).fillna(False)
    
    entries = pd.Series(0, index=df.index)
    entries[entries_long] = 1
    entries[entries_short] = -1
    
    # --- EXIT ---
    # QUAN TRỌNG: Bỏ exit_long khi RSI > 80.
    # Chỉ exit khi gãy cấu trúc (Giá đóng dưới EMA)
    exit_long = (df['Close'] < df['ema']) 
    
    # Exit Short khi giá vá lại EMA
    exit_short = (df['Close'] > df['ema'])
    
    return entries, exit_long, exit_short

def calculate_position_size(df, params):
    """
    Tính toán kích thước position.
    Kích hoạt đòn bẩy (Leverage) khi Trend mạnh để đánh bại Benchmark.
    """
    # Lấy giá trị tham số
    atr_multiplier = params.get('atr_multiplier', 3.0)
    adx_threshold = params.get('adx_threshold', 20)
    
    atr = df['atr'].replace(0, np.nan).fillna(method='ffill')
    close = df['Close']
    
    # Khoảng cách SL theo %
    sl_dist_pct = (atr * atr_multiplier) / close
    sl_dist_pct = sl_dist_pct.replace(0, 0.01)
    
    # Rủi ro chấp nhận cho mỗi trade
    # Tăng nhẹ rủi ro lên vì ta muốn beat benchmark
    risk_per_trade = 0.02 # 3% vốn mỗi lệnh
    
    target_size_pct = risk_per_trade / sl_dist_pct
    
    # Điều chỉnh theo ADX:
    # Nếu ADX rất mạnh (> 30), cho phép đánh gấp đôi (Aggressive scaling)
    adx_factor = np.where(df['adx'] >= (adx_threshold + 10), 2.0, 
                 np.where(df['adx'] >= adx_threshold, 1.5, 1.0))
                 
    target_size_pct = target_size_pct * adx_factor
    
    # --- THAY ĐỔI QUAN TRỌNG: MỞ KHÓA LEVERAGE ---
    # Thay vì clip 0.99, ta cho phép lên tới LEVERAGE (đã khai báo là 5 ở đầu file)
    # Tuy nhiên để an toàn, ta chỉ dùng tối đa 2.0x (200% vốn)
    # Bạn có thể chỉnh số 2.0 thành số khác tùy khẩu vị rủi ro
    
    max_lev_allowed = 2.0 
    final_size = np.clip(target_size_pct, 0.1, max_lev_allowed)
    
    return final_size

def run_vectorbt_backtest(df, params):
    """Chạy backtest với Compounding (Lãi kép) - Đã fix lỗi Reverse"""
    
    # print(f"{Fore.YELLOW}--- ĐANG CHẠY BACKTEST... ---") 
    
    df = calculate_indicators(df.copy(), params)
    entries, exit_long, exit_short = generate_signals(df, params) 
    
    entries_long_np = (entries == 1).values.astype(bool)
    entries_short_np = (entries == -1).values.astype(bool)
    exit_long_np = exit_long.values.astype(bool)
    exit_short_np = exit_short.values.astype(bool)
    
    # Lấy Target % Cash
    size_pct = calculate_position_size(df, params)
    size_np = size_pct.values if isinstance(size_pct, pd.Series) else size_pct
    
    # Stoploss calculation
    atr = df['atr'].values
    close = df['Close'].values
    sl_multiplier = params.get('sl_multiplier', 3.0)
    
    sl_stop_pct = (atr * sl_multiplier) / close
    sl_stop_pct = np.nan_to_num(sl_stop_pct, nan=0.05)

    portfolio = vbt.Portfolio.from_signals(
        close=df['Close'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        
        entries=entries_long_np,      
        exits=exit_long_np,           
        short_entries=entries_short_np, 
        short_exits=exit_short_np,   
        
        size=size_np,        
        size_type='percent', # Giữ nguyên là percent
        
        freq=TIMEFRAME,
        init_cash=CASH,
        fees=COMMISSION,
        slippage=OFFSET_PERCENT,
        
        sl_stop=sl_stop_pct, 
        tp_stop=None, 
        sl_trail=True,        
        
        # --- SỬA LỖI TẠI DÒNG NÀY ---
        # Đổi 'reverse' thành 'close'. 
        # Nghĩa là nếu có tín hiệu ngược chiều, ưu tiên đóng lệnh cũ trước.
        upon_opposite_entry='close',        
        # ----------------------------
        
        accumulate=False
    )
    
    return portfolio, df

def export_vectorbt_results(portfolio, df, filename="vectorbt_results.xlsx"):
    try:
        stats = portfolio.stats()
    except Exception as e:
        print(f"{Fore.RED}⚠️ Lỗi khi tính Portfolio Stats (có thể do freq/data ít): {e}")
     
    trades = portfolio.trades.records_readable
    
    print(f"\n{Fore.GREEN}✅ Backtest hoàn tất!")
    
    # Lấy giá trị từ stats series
    total_return = stats.get('Total Return [%]', 0)
    sharpe_ratio = stats.get('Sharpe Ratio', 0)
    total_trades = stats.get('Total Trades', 0)
    
    print(f"Lợi nhuận: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Số giao dịch: {total_trades}")
    
    if len(trades) == 0:
        print(f"{Fore.RED}Không có giao dịch nào được thực hiện.")
        return
    
    # Loại bỏ cột "Column" nếu tồn tại
    if 'Column' in trades.columns:
        trades = trades.drop(columns=['Column'])
        
    # --- MỚI: TÍNH VÀ KIỂM TRA EXPOSURE PER TRADE ---
    # Exposure = Entry Price * Size
    # Xử lý tên cột linh hoạt vì vectorbt có thể trả về 'Avg Entry Price' hoặc 'Entry Price'
    if 'Avg Entry Price' in trades.columns:
        trades['Exposure'] = trades['Avg Entry Price'] * trades['Size']
    elif 'Entry Price' in trades.columns:
        trades['Exposure'] = trades['Entry Price'] * trades['Size']
    else:
        # Trường hợp hiếm gặp không có giá
        trades['Exposure'] = 0
        
    print(f"\n{Fore.MAGENTA}--- EXPOSURE PER TRADE ANALYSIS ---")
    print(trades['Exposure'].describe().apply(lambda x: format(x, 'f'))) # Format để không bị ra số khoa học
    
    print(f"\n{Fore.CYAN}Top 10 Largest Exposures:")
    # Tạo bản sao để in ấn cho đẹp
    print_cols = ['Entry Time', 'Size', 'Exposure', 'PnL', 'Return']
    # Map tên cột nếu cần thiết để khớp với print_cols
    temp_df = trades.copy()
    if 'Entry Timestamp' in temp_df.columns:
        temp_df = temp_df.rename(columns={'Entry Timestamp': 'Entry Time'})
    
    # Chỉ in các cột tồn tại
    valid_cols = [c for c in print_cols if c in temp_df.columns]
    print(temp_df.sort_values('Exposure', ascending=False).head(10)[valid_cols])
    # -----------------------------------------------
    
    # Xuất ra Excel
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Trades sheet - thêm cột vốn biến động
            if len(trades) > 0:
                # Sắp xếp trades theo thời gian exit
                if 'Exit Timestamp' in trades.columns:
                    trades_sorted = trades.sort_values('Exit Timestamp').copy()
                else:
                    trades_sorted = trades.copy()
                
                # Tính vốn biến động (cumulative PnL)
                trades_sorted['Cumulative_PnL'] = trades_sorted['PnL'].cumsum()
                trades_sorted['Equity'] = CASH + trades_sorted['Cumulative_PnL']
                
                # Đổi tên cột cho dễ đọc
                column_rename = {
                    'Entry Timestamp': 'Entry Time',
                    'Exit Timestamp': 'Exit Time',
                    'Avg Entry Price': 'Entry Price',
                    'Avg Exit Price': 'Exit Price',
                    'PnL': 'PnL',
                    'Return': 'Return',
                    'Size': 'Size',
                    'Direction': 'Direction',
                    'Status': 'Status',
                    'Position Id': 'Position ID'
                }
                
                rename_dict = {k: v for k, v in column_rename.items() if k in trades_sorted.columns}
                if rename_dict:
                    trades_sorted = trades_sorted.rename(columns=rename_dict)
                
                # Chọn các cột để xuất (có thứ tự hợp lý)
                export_cols = []
                # Thêm Exposure vào danh sách xuất
                desired_cols = ['Entry Time', 'Exit Time', 'Size', 'Entry Price', 'Exit Price', 
                           'Exposure', 'PnL', 'Return', 'Direction', 'Status', 'Position ID', 'Equity']
                
                for col in desired_cols:
                    if col in trades_sorted.columns:
                        export_cols.append(col)
                
                if export_cols:
                    trades_sorted[export_cols].to_excel(writer, sheet_name='Trades', index=False)
            
            # Summary sheet - chuyển Series thành DataFrame
            summary_df = pd.DataFrame({
                'Metric': stats.index,
                'Value': stats.values
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Equity curve chi tiết theo thời gian
            equity = portfolio.value()
            equity_df = pd.DataFrame({
                'Time': equity.index,
                'Equity': equity.values
            })
            equity_df.to_excel(writer, sheet_name='Equity_Curve', index=False)
            
            # Thêm sheet Equity_After_Trades để xem vốn sau mỗi trade
            if len(trades) > 0:
                equity_trades_df = pd.DataFrame({
                    'Trade_Number': range(1, len(trades_sorted) + 1),
                    'Exit_Time': trades_sorted['Exit Time'].values if 'Exit Time' in trades_sorted.columns else trades_sorted.index,
                    'PnL': trades_sorted['PnL'].values,
                    'Cumulative_PnL': trades_sorted['Cumulative_PnL'].values,
                    'Equity_After_Trade': trades_sorted['Equity'].values
                })
                equity_trades_df.to_excel(writer, sheet_name='Equity_After_Trades', index=False)
            
        print(f"\n{Fore.GREEN}✅ Xuất Excel thành công: {filename}")
        print(f"{Fore.CYAN}Các sheet trong file:")
        print(f"  - Trades: Danh sách giao dịch (đã thêm cột Exposure)")
        print(f"  - Summary: Thống kê tổng quan")
        print(f"  - Equity_Curve: Đường cong vốn theo thời gian")
        print(f"  - Equity_After_Trades: Vốn biến động sau từng trade")
        
    except Exception as e:
        print(f"{Fore.RED}Lỗi xuất file: {e}")
        traceback.print_exc()

def run_optimization_vectorbt(df):
    print(f"{Fore.CYAN}--- ĐANG TỐI ƯU HÓA THAM SỐ (COMPOUNDING) ---")
    print(f"{Fore.CYAN}Lưu ý: Quá trình này có thể mất vài phút tùy số lượng tổ hợp...")
    
    param_grid = {
        'ema_period': [70, 75, 80, 85],          
        'rsi_period': [20, 21, 22, 23],            
        'atr_multiplier': [2.50, 2.75, 3.0, 3.25, 3.5],      
        'adx_threshold': [14, 15, 16, 17],         
        'sl_multiplier': [6.5, 7.0, 7.5, 8.0]  
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    total_combinations = len(combinations)
    print(f"Tổng số tổ hợp cần kiểm tra: {total_combinations}")
    
    best_return = -float('inf')
    best_params = None
    best_portfolio = None
    
    start_time = time.time()

    for i, combo in enumerate(combinations):
        # Tạo dictionary params từ combo hiện tại
        current_params = dict(zip(keys, combo))
        
        # Log tiến độ mỗi 10 vòng lặp
        if i % 10 == 0 or i == total_combinations - 1:
            elapsed = time.time() - start_time
            print(f"Tiến độ: {i+1}/{total_combinations} | Best Return: {best_return:.2f}% | Time: {elapsed:.0f}s", end='\r')
        
        try:
            # Gọi hàm backtest (Hàm này đã được sửa ở trên)
            # Chúng ta dùng try/except để tránh việc 1 params lỗi làm dừng cả quá trình
            pf, _ = run_vectorbt_backtest(df, current_params)
            
            # Lấy kết quả
            stats = pf.stats()
            total_ret = stats.get('Total Return [%]', -999)
            num_trades = stats.get('Total Trades', 0)
            
            # Điều kiện lọc:
            # 1. Lợi nhuận cao hơn mức tốt nhất hiện tại
            # 2. Số lệnh > 1 (để tránh curve fitting vào 1-2 lệnh may mắn)
            if total_ret > best_return and num_trades > 1:
                best_return = total_ret
                best_params = current_params
                # Không lưu best_portfolio ở đây để tiết kiệm RAM, ta sẽ chạy lại lần cuối
                
        except Exception as e:
            # print(f"Lỗi tại params {current_params}: {e}")
            continue
            
    print(f"\n\n{Fore.GREEN}=== KẾT QUẢ TỐI ƯU HOÀN TẤT ===")
    print(f"Best Total Return: {best_return:.2f}%")
    print(f"Best Params: {best_params}")
    
    if best_params is not None:
        # Chạy lại một lần cuối với tham số tốt nhất để lấy portfolio object
        final_portfolio, _ = run_vectorbt_backtest(df, best_params)
        return best_params, final_portfolio
    
    return None, None

def update_config_file(best_params):
    config_path = 'op_config.json'
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                data = json.load(f)
        else:
            data = {
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "initial_capital": CASH,
                "leverage": str(LEVERAGE),
                "risk_per_trade_percent": MARGIN_PER_TRADE,
                "strategy": {},
                "trailing": {"enabled": True}
            }

        data['strategy']['ema_period'] = int(best_params['ema_period'])
        data['strategy']['rsi_period'] = int(best_params['rsi_period'])
        data['strategy']['atr_multiplier_sl'] = float(best_params['atr_multiplier'])
        data['strategy']['adx_threshold'] = int(best_params['adx_threshold'])
        data['strategy']['sl_multiplier'] = float(best_params['sl_multiplier'])
        
        # --- Cập nhật TP nếu có (hoặc xóa nếu không) ---
        if 'tp_multiplier' in best_params:
            data['strategy']['tp_multiplier'] = float(best_params['tp_multiplier'])
        else:
            data['strategy']['tp_multiplier'] = None # Explicitly set to null in JSON
        
        data['last_optimized'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"{Fore.GREEN}✅ Đã lưu tham số tối ưu vào '{config_path}'!")
    except Exception as e:
        print(f"{Fore.RED}Lỗi cập nhật config: {e}")

if __name__ == "__main__":
    df = fetch_historical_data(SYMBOL, BYBIT_INTERVAL, DAYS_BACK)
    
    print(f"{Fore.YELLOW}--- BACKTEST BAN ĐẦU ---")
    initial_params = {
        'ema_period': 100,
        'rsi_period': 14,
        'atr_multiplier': 3.0,
        'adx_threshold': 30,
        'sl_multiplier': 2.0,  # Thêm mặc định: SL = 2 ATR
        'tp_multiplier': 4.0   # Thêm mặc định: TP = 4 ATR
    }
    
    portfolio, df_with_indicators = run_vectorbt_backtest(df, initial_params)
    export_vectorbt_results(portfolio, df_with_indicators, "vectorbt_initial.xlsx")
    
    # Optimization
    print(f"\n{Fore.CYAN}--- TỐI ƯU HÓA ---")
    
    best_params, best_portfolio = run_optimization_vectorbt(df)
    
    if best_params is not None:
        # Chạy lại với best params
        print(f"{Fore.YELLOW}--- BACKTEST VỚI THAM SỐ TỐI ƯU ---")
        final_portfolio, final_df = run_vectorbt_backtest(df, best_params)
        export_vectorbt_results(final_portfolio, final_df, "vectorbt_optimized.xlsx")
        
        # Lưu config
        update_config_file(best_params)
    else:
        print(f"{Fore.RED}Không tìm thấy tham số tối ưu!")
