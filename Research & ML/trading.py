import os
import json
import time
import math
import logging
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from colorama import Fore, Style, init

# --- 1. KHỞI TẠO ---
init(autoreset=True)
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    with open('config.json', 'r') as f:
        CFG = json.load(f)
except FileNotFoundError:
    logging.error("Thiếu file config.json")
    exit(1)

# Kết nối Bybit
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

DEMO_ENDPOINT = "https://api-demo.bybit.com"

print(f"{Fore.YELLOW}--- KẾT NỐI ĐẾN: DEMO TRADING ---")
print(f"{Fore.CYAN}Target Endpoint: {DEMO_ENDPOINT}")
print(f"Key đang dùng: {api_key[:5]}...{api_key[-4:] if api_key else 'None'}")

try:
    session = HTTP(
        testnet=True,       
        domain="demo",      
        api_key=api_key,
        api_secret=api_secret,
        recv_window=7000,
    )
    session.endpoint = DEMO_ENDPOINT
    print(f"Session Endpoint đã thiết lập: {session.endpoint}")
except Exception as e:
    print(f"Lỗi khởi tạo HTTP session: {e}")
    exit(1)

STATE_FILE = 'bot_state.json'
SYMBOL = CFG['symbol']
CATEGORY = 'linear' # Future

# --- 2. HÀM TIỆN ÍCH ---
def log(msg, color=Fore.WHITE):
    print(f"{color}[{datetime.now().strftime('%H:%M:%S')}] {msg}{Style.RESET_ALL}")

def check_connection():
    try:
        session.get_server_time()
        bal = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        balance = bal['result']['list'][0]['coin'][0]['walletBalance']
        log(f"✅ KẾT NỐI THÀNH CÔNG! Số dư ví DEMO: {float(balance):.2f} USDT", Fore.GREEN)
        return True
    except Exception as e:
        log(f"❌ KẾT NỐI THẤT BẠI: {e}", Fore.RED)
        return False

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"current_r_value": 0, "entry_price": 0}

def save_state(data):
    with open(STATE_FILE, 'w') as f: json.dump(data, f)

def get_precision(symbol):
    try:
        resp = session.get_instruments_info(category=CATEGORY, symbol=symbol)
        price_filter = resp['result']['list'][0]['priceFilter']['tickSize']
        qty_filter = resp['result']['list'][0]['lotSizeFilter']['qtyStep']
        return float(price_filter), float(qty_filter)
    except Exception as e:
        log(f"Lỗi lấy thông tin Symbol: {e}", Fore.RED)
        return 0.1, 0.001

PRICE_PRECISION, QTY_PRECISION = get_precision(SYMBOL)

def round_price(price):
    return "{:.{}f}".format(price, str(PRICE_PRECISION)[::-1].find('.'))

def round_qty(qty):
    steps = int(math.log10(1/QTY_PRECISION))
    return round(qty, steps)

# --- 3. PHÂN TÍCH KỸ THUẬT ---
def fetch_market_data():
    try:
        resp = session.get_kline(category=CATEGORY, symbol=SYMBOL, interval=CFG['timeframe'], limit=300)
        if resp['retCode'] != 0: raise Exception(resp['retMsg'])
        
        data = resp['result']['list']
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'turn'])
        df = df.iloc[::-1].reset_index(drop=True)
        
        cols = ['open', 'high', 'low', 'close']
        df[cols] = df[cols].astype(float)
        
        df['ema'] = df['close'].ewm(span=CFG['strategy']['ema_period'], adjust=False).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=CFG['strategy']['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=CFG['strategy']['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=CFG['strategy']['atr_period']).mean()

        return df.iloc[-2], df.iloc[-1]['close']
    except Exception as e:
        log(f"Lỗi Data: {e}", Fore.RED)
        return None, 0

# --- 4. LOGIC GIAO DỊCH & QUẢN LÝ VỐN ---
def calculate_size(balance, entry, sl):
    risk_amt = balance * CFG['risk_per_trade_percent']
    dist = abs(entry - sl)
    if dist == 0: return 0
    qty = risk_amt / dist
    if qty * entry > CFG['max_position_usdt']:
        qty = CFG['max_position_usdt'] / entry
    return round_qty(qty)

def check_order_status_explicit(order_id):
    """Kiểm tra trạng thái chính xác từ lịch sử lệnh"""
    try:
        # Check history to see if it's Filled or Cancelled
        history = session.get_order_history(category=CATEGORY, symbol=SYMBOL, orderId=order_id, limit=1)
        if history['retCode'] == 0 and history['result']['list']:
            return history['result']['list'][0]['orderStatus'] # 'Filled', 'Cancelled', 'Rejected'
    except Exception as e:
        log(f"Lỗi check history: {e}", Fore.RED)
    return "Unknown"

def check_recent_pnl():
    """Kiểm tra PnL của lệnh vừa đóng"""
    try:
        # Lấy lịch sử closed PnL gần nhất
        resp = session.get_closed_pnl(category=CATEGORY, symbol=SYMBOL, limit=1)
        if resp['retCode'] == 0 and resp['result']['list']:
            recent = resp['result']['list'][0]
            # Kiểm tra thời gian đóng lệnh có gần đây không (trong vòng 60s)
            close_time_ms = int(recent['updatedTime'])
            if (time.time() * 1000) - close_time_ms < 60000: # 60s
                pnl = float(recent['closedPnl'])
                side = recent['side']
                type_close = recent['execType'] # Trade, AdlTrade, SessionSettle...
                
                color = Fore.GREEN if pnl > 0 else Fore.RED
                icon = "💰 CHỐT LỜI" if pnl > 0 else "🛑 CẮT LỖ"
                
                log(f"{icon} ({side}): PnL = {pnl:.2f} USDT | Giá đóng: {recent['avgExitPrice']}", color)
                return
    except Exception as e:
        log(f"Không thể lấy PnL: {e}", Fore.RED)

def execute_trade(side, current_price, atr):
    try:
        bal_resp = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        balance = float(bal_resp['result']['list'][0]['coin'][0]['walletBalance'])
        
        OFFSET_PERCENT = 0.0005 
        if side == 'Buy':
            limit_entry = current_price * (1 - OFFSET_PERCENT)
            sl_price = limit_entry - (atr * CFG['strategy']['atr_multiplier_sl'])
        else:
            limit_entry = current_price * (1 + OFFSET_PERCENT)
            sl_price = limit_entry + (atr * CFG['strategy']['atr_multiplier_sl'])
            
        qty = calculate_size(balance, limit_entry, sl_price)
        if qty <= 0: return

        sl_dist = abs(limit_entry - sl_price)
        log(f"⏳ Đặt chờ {side} LIMIT | Entry: {round_price(limit_entry)} | SL: {round_price(sl_price)} | Qty: {qty}", Fore.YELLOW)
        
        order = session.place_order(
            category=CATEGORY, symbol=SYMBOL, side=side, orderType="Limit",
            qty=str(qty), price=str(round_price(limit_entry)),
            stopLoss=str(round_price(sl_price)),
            timeInForce="GTC"
        )
        
        if order['retCode'] != 0:
            log(f"❌ Lỗi đặt lệnh: {order['retMsg']}", Fore.RED)
            return

        order_id = order['result']['orderId']
        log("... Đang chờ khớp lệnh (Ưu tiên check Position)...", Fore.WHITE)
        
        wait_count = 0
        while True:
            time.sleep(5) 
            wait_count += 1
            
            # Ưu tiên 1: Kiểm tra xem đã có Vị thế (Position) chưa?
            try:
                pos_resp = session.get_positions(category=CATEGORY, symbol=SYMBOL)
                if pos_resp['retCode'] == 0:
                    pos_info = pos_resp['result']['list'][0]
                    if float(pos_info['size']) > 0:
                        save_state({"current_r_value": sl_dist, "entry_price": limit_entry})
                        log(f"✅ KHỚP LỆNH THÀNH CÔNG! Entry: {pos_info['avgPrice']} | Size: {pos_info['size']}", Fore.GREEN)
                        return
            except Exception as e:
                log(f"Lỗi check position: {e}", Fore.RED)

            # Ưu tiên 2: Check Order Status
            ord_status = session.get_open_orders(category=CATEGORY, symbol=SYMBOL, orderId=order_id)
            ticker = session.get_tickers(category=CATEGORY, symbol=SYMBOL)
            curr_market_price = float(ticker['result']['list'][0]['lastPrice'])

            if wait_count % 2 == 0: 
                print(f"\r[Đang chờ] Giá: {curr_market_price} | Entry: {round_price(limit_entry)}...", end="", flush=True)

            is_open = False
            if ord_status['retCode'] == 0:
                if ord_status['result']['list']:
                    is_open = True
            
            if not is_open:
                print("") 
                final_status = check_order_status_explicit(order_id)
                if final_status == 'Cancelled':
                    log("⚠️ Lệnh đã bị HỦY (Thủ công hoặc do sàn).", Fore.RED)
                    return
                elif final_status == 'Rejected':
                    log("❌ Lệnh bị TỪ CHỐI.", Fore.RED)
                    return
            
            # Logic Hủy lệnh 3R
            should_cancel = False
            if side == 'Buy' and curr_market_price > (limit_entry + 3 * sl_dist):
                should_cancel = True
            elif side == 'Sell' and curr_market_price < (limit_entry - 3 * sl_dist):
                should_cancel = True
            
            if should_cancel:
                print("")
                log(f"⚠️ Giá chạy quá xa (3R). Hủy lệnh chờ.", Fore.YELLOW)
                session.cancel_order(category=CATEGORY, symbol=SYMBOL, orderId=order_id)
                return

    except Exception as e:
        log(f"Execution Error: {e}", Fore.RED)

# --- 5. LOGIC TRAILING STOP ---
def manage_position():
    try:
        pos_resp = session.get_positions(category=CATEGORY, symbol=SYMBOL)
        if pos_resp['retCode'] != 0: return
        
        positions = pos_resp['result']['list']
        if not positions or float(positions[0]['size']) == 0:
            return 
            
        pos = positions[0]
        side = pos['side']
        entry = float(pos['avgPrice'])
        current_sl = float(pos['stopLoss']) if pos.get('stopLoss') and pos['stopLoss'] != "" else 0
        mark_price = float(pos['markPrice'])
        
        state = load_state()
        r_value = state.get('current_r_value', 0)
        
        if r_value == 0: 
            r_value = abs(entry - current_sl) if current_sl > 0 else entry * 0.01 
        
        if side == 'Buy':
            current_profit_r = (mark_price - entry) / r_value
        else:
            current_profit_r = (entry - mark_price) / r_value

        if CFG['trailing']['enabled']:
            floor_r = math.floor(current_profit_r)
            target_lock_r = floor_r - 1.0
            
            if target_lock_r >= 0:
                new_sl = None
                if side == 'Buy':
                    calc_sl = entry + (target_lock_r * r_value)
                    if current_sl == 0 or calc_sl > current_sl + (r_value * 0.1): 
                        new_sl = calc_sl
                else:
                    calc_sl = entry - (target_lock_r * r_value)
                    if current_sl == 0 or calc_sl < current_sl - (r_value * 0.1): 
                        new_sl = calc_sl
                
                if new_sl:
                    log(f"💰 TRAILING: Lãi {current_profit_r:.2f}R. Dời SL về {target_lock_r}R ({round_price(new_sl)})", Fore.YELLOW)
                    session.set_trading_stop(
                        category=CATEGORY, symbol=SYMBOL, 
                        stopLoss=str(round_price(new_sl)), positionIdx=0
                    )

    except Exception as e:
        log(f"Lỗi quản lý lệnh: {e}", Fore.RED)

# --- 6. VÒNG LẶP CHÍNH ---
def main():
    log(f"--- BYBIT QUANT ENGINE (v2.3 - FIXED PNL LOGS) STARTED ---", Fore.GREEN)
    
    if not check_connection():
        return

    try:
        session.set_leverage(category=CATEGORY, symbol=SYMBOL, buyLeverage=CFG['leverage'], sellLeverage=CFG['leverage'])
    except: pass 

    # Biến theo dõi trạng thái vòng lặp trước
    was_in_position = False

    while True:
        try:
            # 1. Quản lý lệnh đang chạy (Trailing SL)
            manage_position()
            
            # 2. Kiểm tra trạng thái vị thế
            pos_info = session.get_positions(category=CATEGORY, symbol=SYMBOL)['result']['list'][0]
            current_size = float(pos_info['size'])
            
            if was_in_position and current_size == 0:
                log("ℹ️ Phát hiện vị thế vừa đóng. Kiểm tra PnL...", Fore.CYAN)
                check_recent_pnl()
            
            was_in_position = (current_size > 0)

            # 3. Logic tìm setup mới (Chỉ chạy khi không có lệnh)
            if current_size == 0:
                candle, current_price = fetch_market_data()
                if candle is not None:
                    ema = candle['ema']
                    atr = candle['atr']
                    rsi = candle['rsi']
              
                    c_close = float(candle['close'])
                    c_open = float(candle['open'])
                    
                    if current_price > ema and rsi < 70 and c_close > c_open:
                        execute_trade('Buy', current_price, atr)
                    elif current_price < ema and rsi > 30 and c_close < c_open:
                        execute_trade('Sell', current_price, atr)

            time.sleep(10)

        except KeyboardInterrupt:
            log("🛑 Dừng Bot thủ công.", Fore.YELLOW)
            break
        except Exception as e:
            log(f"CRITICAL ERROR: {e}", Fore.RED)
            time.sleep(5)

if __name__ == "__main__":
    main()
