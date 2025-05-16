import pandas as pd
import requests
import hmac
import hashlib
from datetime import datetime, timezone
from supabase import create_client, Client
import os
import time
from dotenv import load_dotenv
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Binance and Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
BINANCE_API_KEY = os.getenv("APIKey")
BINANCE_API_SECRET = os.getenv("secretKey")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY environment variables must be set")

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET environment variables must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BINANCE_URL = "https://api.binance.com/api/v3/ticker/24hr"

# Constants
INVESTMENT_AMOUNT = 100
BINANCE_FEES_PCT = 0.001
SYMBOL_TTL = 30  # Reduced to check more frequently
recent_symbols = {}

# Global variables for trade tracking
active_trades = {}
TRADE_TIMEOUT = timedelta(minutes=15)  # Reduced timeout for quicker trades
TARGET_PRICE_CHANGE = 0.7  # Target 0.7% price change
MIN_PRICE_CHANGE = 0.05  # Minimum price movement to consider entry
MAX_ACTIVE_TRADES = 3
STOP_LOSS_PCT = -0.05  # Stop loss at -0.05%

def fetch_usdt_tickers():
    try:
        response = requests.get(BINANCE_URL)
        response.raise_for_status()
        data = pd.DataFrame(response.json())
        usdt_tickers = data[data['symbol'].str.endswith("USDT")].copy()
        usdt_tickers['priceChangePercent'] = pd.to_numeric(usdt_tickers['priceChangePercent'], errors='coerce')
        usdt_tickers['quoteVolume'] = pd.to_numeric(usdt_tickers['quoteVolume'], errors='coerce')
        usdt_tickers['timestamp'] = datetime.now(timezone.utc).isoformat()
        return usdt_tickers[['symbol', 'priceChangePercent', 'quoteVolume', 'timestamp']]
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch data from Binance: {e}")
        return pd.DataFrame()

def calculate_score(df):
    if df.empty:
        return df
    df['volatility_score'] = df['priceChangePercent'].abs()
    df['volume_score'] = df['quoteVolume'] / df['quoteVolume'].max()
    df['momentum_score'] = (df['priceChangePercent'] - df['priceChangePercent'].min()) / (
        df['priceChangePercent'].max() - df['priceChangePercent'].min())
    df['composite_score'] = (
        df['volatility_score'] * 0.4 +
        df['volume_score'] * 0.3 +
        df['momentum_score'] * 0.3
    )
    return df[['symbol', 'priceChangePercent', 'quoteVolume', 'composite_score', 'timestamp']].sort_values(by='composite_score', ascending=False)

def save_to_supabase(df):
    for _, row in df.iterrows():
        data = {
            "symbol": row['symbol'],
            "price_change_pct": row['priceChangePercent'],
            "quote_volume": row['quoteVolume'],
            "timestamp": row['timestamp']
        }
        try:
            supabase.table("market_signals").insert(data).execute()
        except Exception as e:
            print(f"[ERROR] Supabase insert failed: {e}")

def calculate_pnl(investment, price_change_pct, fees_pct):
    gross_return = investment * (price_change_pct / 100)
    net_return = gross_return * (1 - fees_pct * 2)
    return round(net_return, 2)

def fetch_15m_price_change(symbol):
    try:
        endpoint = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=2"
        response = requests.get(endpoint)
        response.raise_for_status()
        data = response.json()
        if len(data) >= 2:
            old_price = float(data[0][4])
            new_price = float(data[1][4])
            return round(((new_price - old_price) / old_price) * 100, 2)
        return 0.0
    except Exception as e:
        print(f"[ERROR] Failed to fetch 15m price change for {symbol}: {e}")
        return 0.0

def fetch_5m_price_change(symbol):
    try:
        # Get more candles to ensure we capture actual 5-minute change
        endpoint = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=10"
        response = requests.get(endpoint)
        response.raise_for_status()
        data = response.json()
        
        if len(data) >= 5:  # Need at least 5 one-minute candles
            # Get price from 5 minutes ago and current price
            five_min_ago_price = float(data[-5][4])  # Close price from 5 candles ago
            current_price = float(data[-1][4])       # Most recent close price
            return round(((current_price - five_min_ago_price) / five_min_ago_price) * 100, 2)
        return 0.0
    except Exception as e:
        print(f"[ERROR] Failed to fetch 5m price change for {symbol}: {e}")
        return 0.0

def estimate_prob_3pct_jump_and_price(symbol):
    try:
        endpoint = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=14"
        response = requests.get(endpoint)
        response.raise_for_status()
        data = response.json()
        if len(data) < 14:
            return 0.0, 0.0, 0.0, 0.0

        count_3pct_jumps = 0
        trs = []
        closes = []

        for kline in data:
            high = float(kline[2])
            low = float(kline[3])
            close = float(kline[4])
            open_price = float(kline[1])
            pct_change = ((close - open_price) / open_price) * 100
            closes.append(close)
            trs.append(high - low)
            if pct_change >= 3:
                count_3pct_jumps += 1

        atr = sum(trs) / len(trs)
        current_price = closes[-1]
        prob_3pct = round(count_3pct_jumps / len(data), 3)
        upper_price = round(current_price + atr, 4)
        lower_price = round(current_price - atr, 4)

        return prob_3pct, current_price, upper_price, lower_price

    except Exception as e:
        print(f"[ERROR] Failed to estimate 3% jump probability for {symbol}: {e}")
        return 0.0, 0.0, 0.0, 0.0

def get_spot_balance():
    try:
        timestamp = int(time.time() * 1000)
        params = {
            'timestamp': timestamp
        }
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            BINANCE_API_SECRET.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        url = 'https://api.binance.com/api/v3/account'
        headers = {
            'X-MBX-APIKEY': BINANCE_API_KEY
        }
        params['signature'] = signature
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        account_info = response.json()
        
        # Filter non-zero balances and format them
        balances = []
        for asset in account_info['balances']:
            free = float(asset['free'])
            locked = float(asset['locked'])
            total = free + locked
            if total > 0:
                balances.append({
                    'asset': asset['asset'],
                    'free': free,
                    'locked': locked,
                    'total': total
                })
        
        # Sort by total value
        balances.sort(key=lambda x: x['total'], reverse=True)
        return balances
    except Exception as e:
        print(f"[ERROR] Failed to fetch spot balance: {e}")
        return []

@dataclass
class SimulatedTrade:
    symbol: str
    entry_price: float
    quantity: float
    timestamp: datetime
    probability: float
    initial_pnl: float
    trade_id: Optional[int] = None  # Track the database ID

def save_trade_entry(symbol: str, buy_price: float, quantity: float) -> Optional[int]:
    """Save new trade to Supabase and return the trade ID"""
    try:
        data = {
            "symbol": symbol,
            "buy_price": buy_price,
            "buy_time": datetime.now(timezone.utc).isoformat(),
            "quantity": quantity
        }
        result = supabase.table("trades").insert(data).execute()
        return result.data[0]["id"]
    except Exception as e:
        print(f"[ERROR] Failed to save trade entry to Supabase: {e}")
        return None

def update_trade_exit(trade_id: int, sell_price: float, profit_loss: float):
    """Update trade in Supabase with exit information"""
    try:
        data = {
            "sell_price": sell_price,
            "sell_time": datetime.now(timezone.utc).isoformat(),
            "profit_loss": profit_loss
        }
        supabase.table("trades").update(data).eq("id", trade_id).execute()
    except Exception as e:
        print(f"[ERROR] Failed to update trade exit in Supabase: {e}")

# Removed duplicate trade functions as we're using simulate_trade_entry and check_trade_exit instead

def evaluate_trading_opportunity(symbol: str, delta_15m: float, delta_5m: float, prob_3pct: float, 
                               current_price: float, pnl: float):
    """Evaluate if we should enter a trade based on price movements"""
    # Check for minimum price movement in last 5 minutes
    if delta_5m < MIN_PRICE_CHANGE:
        return
        
    # Calculate quantity based on investment amount
    quantity = INVESTMENT_AMOUNT / current_price
    
    # If criteria met, simulate trade entry with 5m change as initial_pnl
    if simulate_trade_entry(symbol, current_price, quantity, prob_3pct, delta_5m):
        print(f"[INFO] Started monitoring trade for {symbol}")
        
def check_trade_exit(symbol: str, current_price: float) -> bool:
    """Check if we should exit a trade based on profit target, stop loss, or timeout"""
    if symbol not in active_trades:
        return False
        
    trade = active_trades[symbol]
    current_time = datetime.now()
    price_change = ((current_price - trade.entry_price) / trade.entry_price) * 100
    actual_pnl = calculate_pnl(INVESTMENT_AMOUNT, price_change, BINANCE_FEES_PCT)
    
    timeout_exit = (current_time - trade.timestamp) > TRADE_TIMEOUT
    profit_target_reached = price_change >= TARGET_PRICE_CHANGE
    stop_loss_triggered = price_change <= STOP_LOSS_PCT
    if timeout_exit or profit_target_reached or stop_loss_triggered:
        exit_reason = "Profit Target ✅" if profit_target_reached else (
            "Stop Loss ❌" if stop_loss_triggered else "Timeout ⏰"
        )
        
        print(f"\n[TRADE EXIT] {symbol}")
        print(f"Entry Price: {trade.entry_price:.8f}")
        print(f"Exit Price: {current_price:.8f} ({price_change:+.2f}%)")
        print(f"Time in trade: {current_time - trade.timestamp}")
        print(f"Investment: {INVESTMENT_AMOUNT:.2f} USDT")
        print(f"Final PnL: {actual_pnl:.2f} USDT")
        print(f"Reason: {exit_reason}")
        print(f"Fees paid: {(INVESTMENT_AMOUNT * BINANCE_FEES_PCT * 2):.4f} USDT")
        
        # Update trade exit in Supabase
        if trade.trade_id is not None:
            update_trade_exit(trade.trade_id, current_price, actual_pnl)
        
        del active_trades[symbol]
        return True
    return False

def simulate_trade_entry(symbol: str, current_price: float, quantity: float, probability: float, initial_pnl: float):
    """Simulate entering a trade"""
    if len(active_trades) >= MAX_ACTIVE_TRADES:
        return False
        
    if symbol in active_trades:
        return False
    
    # Save trade to database first
    trade_id = save_trade_entry(symbol, current_price, quantity)
        
    trade = SimulatedTrade(
        symbol=symbol,
        entry_price=current_price,
        quantity=quantity,
        timestamp=datetime.now(),
        probability=probability,
        initial_pnl=initial_pnl,
        trade_id=trade_id  # Store the database ID
    )
    active_trades[symbol] = trade
    
    print(f"\n[TRADE ENTRY] {symbol}")
    print(f"Entry Price: {current_price:.8f}")
    print(f"Quantity: {quantity:.8f}")
    print(f"Initial PnL: {initial_pnl:.2f}%")
    print(f"Probability: {probability:.3f}")
    if trade_id:
        print(f"Trade ID: {trade_id}")
    return True

def main():
    # Print spot wallet balance at startup
    print("\n=== Spot Wallet Balance ===")
    balances = get_spot_balance()
    if balances:
        for balance in balances:
            print(f"{balance['asset']}: Free={balance['free']:.8f}, Locked={balance['locked']:.8f}, Total={balance['total']:.8f}")
    print("=========================\n")
    
    while True:
        df = fetch_usdt_tickers()
        if df.empty:
            print("[INFO] No data returned from Binance API.")
            time.sleep(1)
            continue

        scored_df = calculate_score(df)
        filtered_df = scored_df.head(10)
        now = time.time()

        for _, row in filtered_df.iterrows():
            symbol = row['symbol']
            if symbol in recent_symbols and (now - recent_symbols[symbol]) < SYMBOL_TTL:
                continue

            recent_symbols[symbol] = now
            delta_15m = fetch_15m_price_change(symbol)
            delta_5m = fetch_5m_price_change(symbol)
            prob_3pct, current_price, upper_price, lower_price = estimate_prob_3pct_jump_and_price(symbol)

            if delta_15m == 0.0 or delta_5m == 0.0:
                continue

            pnl = calculate_pnl(INVESTMENT_AMOUNT, delta_15m, BINANCE_FEES_PCT)            # Print market data
            if pnl > 0:
                print(f"📈 {symbol}: ∆15m={delta_15m}%, ∆5m={delta_5m}%, Prob≥3%={prob_3pct}, Vol={row['quoteVolume']:.2f}, "
                      f"PnL={pnl:.2f} USDT, Price={current_price}, Range=({lower_price} - {upper_price})")

            # Check existing trades
            if symbol in active_trades:
                check_trade_exit(symbol, current_price)
            # Evaluate new trading opportunities
            elif pnl > 0:
                evaluate_trading_opportunity(symbol, delta_15m, delta_5m, prob_3pct, current_price, pnl)            # Save to database
            save_to_supabase(pd.DataFrame([{
                'symbol': symbol,
                'priceChangePercent': delta_15m,
                'quoteVolume': row['quoteVolume'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }]))

        # Sleep briefly between iterations
        time.sleep(1)

if __name__ == "__main__":
    main()
