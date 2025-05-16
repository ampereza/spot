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
SYMBOL_TTL = 60  # Increased to avoid overtrading
recent_symbols = {}

# Trading Parameters
MIN_15M_CHANGE = 1.0     # Minimum 15m price change to consider entry
MIN_5M_CHANGE = 0.3      # Minimum 5m price change to consider entry
MIN_PROBABILITY = 0.15    # Minimum probability of 3% jump
MIN_VOLUME_PERCENTILE = 75  # Only trade coins in top 75% by volume

# Global variables for trade tracking
active_trades = {}
MAX_ACTIVE_TRADES = 2     # Reduced to manage risk better
TRADE_TIMEOUT = timedelta(minutes=30)  # Increased to give trades more room

# Blacklist settings
symbol_blacklist = {}
BLACKLIST_DURATION = 1800  # 30-minute cooldown after a loss
consecutive_losses = {}    # Track consecutive losses per symbol
MAX_CONSECUTIVE_LOSSES = 2  # Blacklist after 2 consecutive losses

# Performance tracking
trade_history = []
MAX_TRADE_HISTORY = 20    # Keep last 20 trades for statistics

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
    stop_loss: float = 0.0
    take_profit: float = 0.0
    volume_percentile: float = 0.0
    atr: float = 0.0

@dataclass
class TradeResult:
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    duration: timedelta
    exit_reason: str
    timestamp: datetime

class TradeStats:
    def __init__(self, max_history: int = 20):
        self.trades = []
        self.max_history = max_history
    
    def add_trade(self, trade: TradeResult):
        self.trades.append(trade)
        if len(self.trades) > self.max_history:
            self.trades.pop(0)
    
    def get_win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)
    
    def get_avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl for t in self.trades) / len(self.trades)
    
    def get_max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        cumulative = 0
        max_drawdown = 0
        peak = 0
        
        for trade in self.trades:
            cumulative += trade.pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            max_drawdown = min(max_drawdown, drawdown)
        
        return max_drawdown

class RiskManager:
    def __init__(self):
        self.blacklist = {}
        self.consecutive_losses = {}
    
    def should_blacklist(self, symbol: str, pnl: float) -> bool:
        # Update consecutive losses
        if pnl <= 0:
            self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
            if self.consecutive_losses[symbol] >= MAX_CONSECUTIVE_LOSSES:
                self.blacklist[symbol] = time.time() + BLACKLIST_DURATION
                return True
        else:
            self.consecutive_losses[symbol] = 0
        return False
    
    def is_blacklisted(self, symbol: str) -> bool:
        if symbol in self.blacklist:
            if time.time() > self.blacklist[symbol]:
                del self.blacklist[symbol]
                return False
            return True
        return False

    def calculate_dynamic_limits(self, symbol: str, entry_price: float, atr: float) -> tuple[float, float]:
        """Calculate dynamic stop loss and take profit based on ATR"""
        stop_loss = entry_price - (1.5 * atr)
        take_profit = entry_price + (2.0 * atr)
        return stop_loss, take_profit

# Initialize global instances
trade_stats = TradeStats(MAX_TRADE_HISTORY)
risk_manager = RiskManager()

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
                               current_price: float, pnl: float, volume_percentile: float, atr: float):
    """Evaluate if we should enter a trade based on enhanced criteria"""
    # Check if symbol is blacklisted
    if risk_manager.is_blacklisted(symbol):
        return

    # Enhanced entry criteria
    if (delta_15m < MIN_15M_CHANGE or
        delta_5m < MIN_5M_CHANGE or
        prob_3pct < MIN_PROBABILITY or
        volume_percentile < MIN_VOLUME_PERCENTILE):
        return
        
    # Calculate quantity based on investment amount
    quantity = INVESTMENT_AMOUNT / current_price
    
    # Calculate dynamic stop loss and take profit levels
    stop_loss, take_profit = risk_manager.calculate_dynamic_limits(symbol, current_price, atr)
    
    # If criteria met, simulate trade entry
    if simulate_trade_entry(symbol, current_price, quantity, prob_3pct, delta_5m, stop_loss, take_profit):
        print(f"[INFO] Started monitoring trade for {symbol}")
        
def check_trade_exit(symbol: str, current_price: float) -> bool:
    """Check if we should exit a trade based on dynamic limits"""
    if symbol not in active_trades:
        return False
        
    trade = active_trades[symbol]
    current_time = datetime.now()
    price_change = ((current_price - trade.entry_price) / trade.entry_price) * 100
    actual_pnl = calculate_pnl(INVESTMENT_AMOUNT, price_change, BINANCE_FEES_PCT)
    
    timeout_exit = (current_time - trade.timestamp) > TRADE_TIMEOUT
    profit_target_reached = current_price >= trade.take_profit
    stop_loss_triggered = current_price <= trade.stop_loss
    
    if timeout_exit or profit_target_reached or stop_loss_triggered:
        exit_reason = "Profit Target ✅" if profit_target_reached else (
            "Stop Loss ❌" if stop_loss_triggered else "Timeout ⏰"
        )
        
        trade_duration = current_time - trade.timestamp
        
        print(f"\n[TRADE EXIT] {symbol}")
        print(f"Entry Price: {trade.entry_price:.8f}")
        print(f"Exit Price: {current_price:.8f} ({price_change:+.2f}%)")
        print(f"Time in trade: {trade_duration}")
        print(f"Investment: {INVESTMENT_AMOUNT:.2f} USDT")
        print(f"Final PnL: {actual_pnl:.2f} USDT")
        print(f"Reason: {exit_reason}")
        print(f"Fees paid: {(INVESTMENT_AMOUNT * BINANCE_FEES_PCT * 2):.4f} USDT")
        
        # Record trade result
        record_trade_result(
            symbol=symbol,
            entry_price=trade.entry_price,
            exit_price=current_price,
            quantity=trade.quantity,
            pnl=actual_pnl,
            duration=trade_duration,
            exit_reason=exit_reason
        )
        
        # Update trade exit in Supabase
        if trade.trade_id is not None:
            update_trade_exit(trade.trade_id, current_price, actual_pnl)
        
        del active_trades[symbol]
        return True
    return False

def calculate_volume_percentile(volume: float, all_volumes: List[float]) -> float:
    """Calculate the percentile rank of a volume among all volumes"""
    if not all_volumes:
        return 0.0
    return (sum(1 for v in all_volumes if v <= volume) / len(all_volumes)) * 100

def simulate_trade_entry(symbol: str, current_price: float, quantity: float, probability: float, 
                        initial_pnl: float, stop_loss: float, take_profit: float, 
                        volume_percentile: float = 0.0, atr: float = 0.0):
    """Simulate entering a trade with enhanced parameters"""
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
        trade_id=trade_id,
        stop_loss=stop_loss,
        take_profit=take_profit,
        volume_percentile=volume_percentile,
        atr=atr
    )
    active_trades[symbol] = trade
    
    print(f"\n[TRADE ENTRY] {symbol}")
    print(f"Entry Price: {current_price:.8f}")
    print(f"Stop Loss: {stop_loss:.8f} ({((stop_loss - current_price) / current_price * 100):.2f}%)")
    print(f"Take Profit: {take_profit:.8f} ({((take_profit - current_price) / current_price * 100):.2f}%)")
    print(f"Quantity: {quantity:.8f}")
    print(f"Initial PnL: {initial_pnl:.2f}%")
    print(f"Probability: {probability:.3f}")
    print(f"Volume Percentile: {volume_percentile:.1f}")
    if trade_id:
        print(f"Trade ID: {trade_id}")
    return True

def display_performance_stats():
    """Display current trading performance statistics"""
    win_rate = trade_stats.get_win_rate()
    avg_pnl = trade_stats.get_avg_pnl()
    max_drawdown = trade_stats.get_max_drawdown()
    
    # Calculate top performing symbols
    symbol_pnl = {}
    for trade in trade_stats.trades:
        symbol_pnl[trade.symbol] = symbol_pnl.get(trade.symbol, 0) + trade.pnl
    
    top_symbols = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("\n=== Performance Dashboard ===")
    print(f"Win Rate (last {MAX_TRADE_HISTORY} trades): {win_rate:.1%}")
    print(f"Average PnL: {avg_pnl:.2f} USDT")
    print(f"Max Drawdown: {max_drawdown:.2f} USDT")
    print("\nTop Performing Symbols:")
    for symbol, pnl in top_symbols:
        print(f"  {symbol}: {pnl:+.2f} USDT")
    print(f"\nActive Trades: {len(active_trades)}/{MAX_ACTIVE_TRADES}")
    print(f"Blacklisted Symbols: {len(risk_manager.blacklist)}")
    print("==========================\n")

def record_trade_result(symbol: str, entry_price: float, exit_price: float, 
                       quantity: float, pnl: float, duration: timedelta, 
                       exit_reason: str):
    """Record a completed trade and update statistics"""
    trade_result = TradeResult(
        symbol=symbol,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        pnl=pnl,
        duration=duration,
        exit_reason=exit_reason,
        timestamp=datetime.now(timezone.utc)
    )
    trade_stats.add_trade(trade_result)
    
    # Check if symbol should be blacklisted
    if risk_manager.should_blacklist(symbol, pnl):
        print(f"⛔ {symbol} blacklisted for {BLACKLIST_DURATION/60:.0f} minutes due to consecutive losses")

def main():
    last_stats_display = 0
    STATS_DISPLAY_INTERVAL = 300  # Show stats every 5 minutes
    
    # Print spot wallet balance at startup
    print("\n=== Spot Wallet Balance ===")
    balances = get_spot_balance()
    if balances:
        for balance in balances:
            print(f"{balance['asset']}: Free={balance['free']:.8f}, Locked={balance['locked']:.8f}, Total={balance['total']:.8f}")
    print("=========================\n")
    
    # Display initial performance dashboard
    display_performance_stats()
    
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
                      f"PnL={pnl:.2f} USDT, Price={current_price}, Range=({lower_price} - {upper_price})")            # Calculate volume percentile for this symbol
            volume_percentile = calculate_volume_percentile(
                row['quoteVolume'],
                filtered_df['quoteVolume'].tolist()
            )
            
            # Use ATR from probability calculation
            _, _, upper_price, lower_price = estimate_prob_3pct_jump_and_price(symbol)
            atr = upper_price - lower_price if upper_price and lower_price else 0.0
            
            # Check existing trades
            if symbol in active_trades:
                check_trade_exit(symbol, current_price)
            # Evaluate new trading opportunities
            elif pnl > 0:
                evaluate_trading_opportunity(
                    symbol, delta_15m, delta_5m, prob_3pct, 
                    current_price, pnl, volume_percentile, atr
                )
            save_to_supabase(pd.DataFrame([{
                'symbol': symbol,
                'priceChangePercent': delta_15m,
                'quoteVolume': row['quoteVolume'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }]))        # Display performance stats periodically
        current_time = time.time()
        if current_time - last_stats_display >= STATS_DISPLAY_INTERVAL:
            display_performance_stats()
            last_stats_display = current_time
        
        # Sleep briefly between iterations
        time.sleep(1)

if __name__ == "__main__":
    main()
