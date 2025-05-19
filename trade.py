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
INITIAL_INVESTMENT = 100  # Initial investment amount
BINANCE_FEES_PCT = 0.00075  # 0.075% when paying with BNB
SYMBOL_TTL = 5  # Reduced to check more frequently (5 seconds)
DEBUG_MODE = True  # Enable debug logging

# Global variables for trade tracking
active_trades = {}
TRADE_TIMEOUT = timedelta(minutes=30)
TARGET_PRICE_CHANGE = 0.35  # Target 0.35% price change for taking profits
LIVE_PRICE_CHANGE_MIN = 0.1  # Minimum 0.5% live price change to consider entry
STOP_LOSS_PCT = -1.0  # Stop loss at 1% loss
MIN_PRICE_CHANGE = 0.05  # Minimum price movement to consider entry
MAX_ACTIVE_TRADES = 3
HIGH_PROB_THRESHOLD = 0.5  # 50% probability threshold for high probability trades

def log_debug(msg: str):
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")

def fetch_usdt_tickers():
    try:
        log_debug("Fetching USDT tickers...")
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
    trading_fees = investment * fees_pct * 2  # Entry and exit fees (0.075% * 2 = 0.15% total)
    trend_fee = investment * 0.001  # 0.1% trend fee
    net_return = gross_return - trading_fees - trend_fee
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

def update_trade_exit(trade_id: int, sell_price: float, profit_loss: float, exit_reason: str):
    """Update trade in Supabase with exit information including reason"""
    try:
        data = {
            "sell_price": sell_price,
            "sell_time": datetime.now(timezone.utc).isoformat(),
            "profit_loss": profit_loss,
            "reason": exit_reason
        }
        supabase.table("trades").update(data).eq("id", trade_id).execute()
    except Exception as e:
        print(f"[ERROR] Failed to update trade exit in Supabase: {e}")

def fetch_active_trades_from_db():
    """Fetch active trades from database that haven't been closed"""
    try:
        result = supabase.table("trades").select("*").is_("sell_time", "null").execute()
        recovered_trades = {}
        for trade in result.data:
            recovered_trades[trade['symbol']] = SimulatedTrade(
                symbol=trade['symbol'],
                entry_price=trade['buy_price'],
                quantity=trade['quantity'],
                timestamp=datetime.fromisoformat(trade['buy_time'].replace('Z', '+00:00')),
                probability=0.0,  # Default value since we don't store this
                initial_pnl=0.0,  # Default value since we don't store this
                trade_id=trade['id']
            )
        return recovered_trades
    except Exception as e:
        print(f"[ERROR] Failed to fetch active trades from database: {e}")
        return {}

def get_current_probability(symbol: str) -> float:
    """Get current probability for a symbol"""
    prob, _, _, _ = estimate_prob_3pct_jump_and_price(symbol)
    return prob

def force_exit_lowest_probability_trade() -> bool:
    """Force exit the trade with lowest current probability"""
    if not active_trades:
        return False
        
    lowest_prob_symbol = None
    lowest_prob = float('inf')
    
    # Find trade with lowest current probability
    for symbol in active_trades:
        current_prob = get_current_probability(symbol)
        if current_prob < lowest_prob:
            lowest_prob = current_prob
            lowest_prob_symbol = symbol
    
    if lowest_prob_symbol and lowest_prob < HIGH_PROB_THRESHOLD:
        # Get current price for exit
        _, current_price, _, _ = estimate_prob_3pct_jump_and_price(lowest_prob_symbol)
        if current_price > 0:
            print(f"[TRADE REPLACEMENT] Exiting low probability trade {lowest_prob_symbol} (prob={lowest_prob:.2f})")
            return check_trade_exit(lowest_prob_symbol, current_price)
    return False

def evaluate_trading_opportunity(symbol: str, delta_15m: float, delta_5m: float, prob_3pct: float, 
                               current_price: float, pnl: float):
    """Evaluate if we should enter a trade based on price movements and probability"""
    # Validate current price
    if not current_price or current_price <= 0:
        print(f"[WARNING] Invalid price for {symbol}: {current_price}")
        return
        
    # Check for minimum price movement in last 5 minutes
    if delta_5m < MIN_PRICE_CHANGE:
        return

    # If this is a high probability trade and we're at max capacity
    if prob_3pct >= HIGH_PROB_THRESHOLD and len(active_trades) >= MAX_ACTIVE_TRADES:
        # Try to exit a lower probability trade
        if force_exit_lowest_probability_trade():
            print(f"[HIGH PROBABILITY] Made room for high probability trade {symbol}")
        else:
            return

    # Skip if we already have maximum trades and this is not a high probability trade
    if len(active_trades) >= MAX_ACTIVE_TRADES and prob_3pct < HIGH_PROB_THRESHOLD:
        return
        
    # Check if trade exists in database but not in memory
    try:
        result = supabase.table("trades").select("*").eq("symbol", symbol).is_("sell_time", "null").execute()
        if result.data:
            trade_data = result.data[0]
            # Recover trade to memory if found
            active_trades[symbol] = SimulatedTrade(
                symbol=symbol,
                entry_price=trade_data['buy_price'],
                quantity=trade_data['quantity'],
                timestamp=datetime.fromisoformat(trade_data['buy_time'].replace('Z', '+00:00')),
                probability=0.0,
                initial_pnl=0.0,
                trade_id=trade_data['id']
            )
            print(f"[INFO] Recovered existing trade for {symbol} from database")
            return
    except Exception as e:
        print(f"[ERROR] Failed to check existing trade in database: {e}")
        
    # Calculate quantity based on investment amount
    quantity = INVESTMENT_AMOUNT / current_price
    
    # Higher investment for high probability trades
    if prob_3pct >= HIGH_PROB_THRESHOLD:
        quantity *= 1.5  # Increase position size by 50% for high probability trades
        print(f"[HIGH PROBABILITY] Found high probability trade for {symbol} (prob={prob_3pct:.2f})")
    
    # If criteria met, simulate trade entry
    if simulate_trade_entry(symbol, current_price, quantity, prob_3pct, delta_5m):
        print(f"[INFO] Started monitoring trade for {symbol}")

def check_trade_exit(symbol: str, current_price: float) -> bool:
    """Check if we should exit a trade based on real-time price"""
    if symbol not in active_trades:
        return False
        
    trade = active_trades[symbol]
    current_time = datetime.now(timezone.utc)
    
    # Get real-time price from price tracker - prioritize real-time price
    realtime_price = price_tracker.get_price(symbol)
    if realtime_price:  # Use real-time price if available
        current_price = realtime_price
    
    # Skip if price is invalid
    if not current_price or current_price <= 0:
        print(f"[WARNING] Invalid exit price for {symbol}: {current_price}")
        return False
    
    price_change = ((current_price - trade.entry_price) / trade.entry_price) * 100
    actual_pnl = calculate_pnl(INVESTMENT_AMOUNT, price_change, BINANCE_FEES_PCT)
    
    # Quick exit conditions - Exit immediately if price is at or above entry
    quick_exit = current_price >= trade.entry_price  # Immediate exit on breakeven or profit
    timeout_exit = (current_time - trade.timestamp) > TRADE_TIMEOUT
    stop_loss_triggered = price_change <= STOP_LOSS_PCT  # Keep stop loss for protection
    profit_target_reached = price_change >= TARGET_PRICE_CHANGE
    
    # Prioritize quick exit if price is favorable
    if quick_exit:
        exit_reason = "Quick Exit ⚡"
        print(f"\n[TRADE EXIT] {symbol}")
        print(f"Entry Price: {trade.entry_price:.8f}")
        print(f"Exit Price: {current_price:.8f} ({price_change:+.2f}%)")
        print(f"Time in trade: {current_time - trade.timestamp}")
        print(f"Investment: {INITIAL_INVESTMENT:.2f} USDT")
        print(f"Final PnL: {actual_pnl:.2f} USDT")
        print(f"Reason: {exit_reason}")
        
        if trade.trade_id is not None:
            update_trade_exit(trade.trade_id, current_price, actual_pnl, exit_reason)
        
        del active_trades[symbol]
        return True
        
    if timeout_exit or profit_target_reached or stop_loss_triggered:
        if profit_target_reached:
            exit_reason = "Profit Target ✅"
        elif stop_loss_triggered:
            exit_reason = "Stop Loss ⛔"
        else:
            exit_reason = "Timeout ⏰"
        
        print(f"\n[TRADE EXIT] {symbol}")
        print(f"Entry Price: {trade.entry_price:.8f}")
        print(f"Exit Price: {current_price:.8f} ({price_change:+.2f}%)")
        print(f"Time in trade: {current_time - trade.timestamp}")
        print(f"Investment: {INVESTMENT_AMOUNT:.2f} USDT")
        print(f"Final PnL: {actual_pnl:.2f} USDT")
        print(f"Reason: {exit_reason}")
        print(f"Fees paid: {(INVESTMENT_AMOUNT * BINANCE_FEES_PCT * 2):.4f} USDT")
        
        # Update trade exit in Supabase with reason
        if trade.trade_id is not None:
            update_trade_exit(trade.trade_id, current_price, actual_pnl, exit_reason)
        
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
        timestamp=datetime.now(timezone.utc),  # Make timezone-aware
        probability=probability,
        initial_pnl=initial_pnl,
        trade_id=trade_id
    )
    active_trades[symbol] = trade
    
    # Log the simulated trade entry
    print(f"[TRADE ENTRY] {symbol}")
    print(f"Entry Price: {current_price:.8f}")
    print(f"Quantity: {quantity:.4f}")
    print(f"Probability of 3% jump: {probability:.2f}%")
    print(f"Initial PnL (5m change): {initial_pnl:.2f} USDT")
    print(f"Trade ID: {trade_id}")
    print("===============================")
    
    return True

def fetch_price_changes(symbol: str) -> Dict[str, float]:
    """Fetch price changes for multiple timeframes"""
    timeframes = {
        '24h': {'interval': '1d', 'limit': 2},
        '6h': {'interval': '6h', 'limit': 2},
        '1h': {'interval': '1h', 'limit': 2},
        '30m': {'interval': '30m', 'limit': 2},
        '15m': {'interval': '15m', 'limit': 2},
        '5m': {'interval': '5m', 'limit': 2},
        'live': {'interval': '1m', 'limit': 1}
    }
    
    changes = {}
    
    for tf, params in timeframes.items():
        try:
            endpoint = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={params['interval']}&limit={params['limit']}"
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()
            
            if tf == 'live' and len(data) >= 1:
                # For live price, compare open and current price of the current candle
                open_price = float(data[0][1])
                current_price = float(data[0][4])
                changes[tf] = round(((current_price - open_price) / open_price) * 100, 2)
            elif len(data) >= 2:
                # For other timeframes, compare previous close with current close
                old_price = float(data[0][4])
                new_price = float(data[1][4])
                changes[tf] = round(((new_price - old_price) / old_price) * 100, 2)
            else:
                changes[tf] = 0.0
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch {tf} price change for {symbol}: {e}")
            changes[tf] = 0.0
    
    return changes

def get_total_capital():
    """Calculate total capital based on initial investment plus profits/losses"""
    try:
        # Query all completed trades
        result = supabase.table("trades").select("profit_loss").not_.is_("sell_time", "null").execute()
        if result.data:
            total_pnl = sum(trade['profit_loss'] or 0 for trade in result.data)  # Handle None values
            total_capital = INITIAL_INVESTMENT + total_pnl
            # Set investment amount to 95% of total capital
            return round(total_capital * 0.95, 2)
        return INITIAL_INVESTMENT  # Return initial investment if no trades found
    except Exception as e:
        print(f"[ERROR] Failed to calculate total capital: {e}")
        return INITIAL_INVESTMENT

def get_trading_stats(days: int = 7):
    """Get trading statistics for the specified number of days"""
    try:
        # Calculate the start date
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        # Query completed trades within the time period
        result = supabase.table("trades").select("*").gte("sell_time", start_date).execute()
        
        if not result.data:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_profit_per_trade': 0
            }
        
        trades = result.data
        winning_trades = sum(1 for t in trades if t['profit_loss'] and t['profit_loss'] > 0)
        total_pnl = sum(t['profit_loss'] or 0 for t in trades)
        
        stats = {
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'win_rate': round(winning_trades / len(trades) * 100, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_profit_per_trade': round(total_pnl / len(trades), 2)
        }
        return stats
    except Exception as e:
        print(f"[ERROR] Failed to calculate trading stats: {e}")
        return None

def get_dynamic_position_size(current_capital: float, symbol: str) -> float:
    """Calculate position size based on win rate and volatility"""
    try:
        # Get recent trading performance
        stats = get_trading_stats(7)  # Last 7 days
        if not stats:
            return current_capital * 0.95  # Default to 95% if no stats
        
        # Base position size on win rate
        if stats['win_rate'] >= 60:
            position_size = current_capital * 0.95  # More aggressive when win rate is high
        elif stats['win_rate'] >= 50:
            position_size = current_capital * 0.80  # Moderate when win rate is decent
        else:
            position_size = current_capital * 0.60  # Conservative when win rate is low
        
        # Adjust for recent performance trend
        if stats['total_pnl'] < 0:
            position_size *= 0.8  # Reduce position size after losses
        
        # Round to 2 decimal places
        return round(position_size, 2)
    except Exception as e:
        print(f"[ERROR] Failed to calculate dynamic position size: {e}")
        return current_capital * 0.95  # Default to 95% if error

def calculate_compound_growth(initial_capital: float = INITIAL_INVESTMENT, trades_per_day: int = 50):
    """Calculate compound growth projections based on historical performance"""
    try:
        # Get recent performance stats
        stats = get_trading_stats(7)
        if not stats or stats['total_trades'] == 0:
            return None

        # Calculate average profit percentage per trade (accounting for fees and actual trade PnL)
        avg_profit_pct = (stats['total_pnl'] / (INITIAL_INVESTMENT * stats['total_trades'])) * 100

        # Daily compound calculation (compound each trade)
        daily_trades = trades_per_day
        daily_growth = initial_capital * pow((1 + avg_profit_pct/100), daily_trades)
        daily_return_pct = ((daily_growth - initial_capital) / initial_capital) * 100
        
        # Weekly compound calculation (5 trading days)
        weekly_trades = daily_trades * 7
        weekly_growth = initial_capital * pow((1 + avg_profit_pct/100), weekly_trades)
        weekly_return_pct = ((weekly_growth - initial_capital) / initial_capital) * 100
        
        # Monthly compound calculation (21 trading days)
        monthly_trades = daily_trades * 30
        monthly_growth = initial_capital * pow((1 + avg_profit_pct/100), monthly_trades)
        monthly_return_pct = ((monthly_growth - initial_capital) / initial_capital) * 100
        
        return {
            'avg_profit_per_trade_pct': round(avg_profit_pct, 4),
            'daily': {
                'trades': daily_trades,
                'profit_pct': round(daily_return_pct, 2),
                'projected_growth': round(daily_growth, 2)
            },
            'weekly': {
                'trades': weekly_trades,
                'profit_pct': round(weekly_return_pct, 2),
                'projected_growth': round(weekly_growth, 2)
            },
            'monthly': {
                'trades': monthly_trades,
                'profit_pct': round(monthly_return_pct, 2),
                'projected_growth': round(monthly_growth, 2)
            }
        }
    except Exception as e:
        print(f"[ERROR] Failed to calculate compound growth: {e}")
        return None

def generate_daily_report():
    """Generate a daily trading report and save it to a file"""
    try:
        # Get today's date in UTC
        today = datetime.now(timezone.utc).date()
        report_filename = f"report_{today.strftime('%Y_%m_%d')}.txt"
        
        # Get today's trades
        start_date = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc).isoformat()
        end_date = datetime.combine(today, datetime.max.time()).replace(tzinfo=timezone.utc).isoformat()
        
        result = supabase.table("trades").select("*").gte("buy_time", start_date).lte("buy_time", end_date).execute()
        
        # Get opening capital (total capital at start of day)
        opening_capital = get_total_capital()
        
        if result.data:
            trades = result.data
            total_trades = len(trades)
            completed_trades = [t for t in trades if t['sell_time'] is not None]
            total_pnl = sum(t['profit_loss'] or 0 for t in completed_trades)
            avg_pnl = total_pnl / len(completed_trades) if completed_trades else 0
            
            # Generate report content
            report_content = f"=== Daily Trading Report {today.strftime('%Y-%m-%d')} ===\n\n"
            report_content += f"Opening Capital: {opening_capital:.2f} USDT\n"
            report_content += f"Total Trades: {total_trades}\n"
            report_content += f"Completed Trades: {len(completed_trades)}\n"
            report_content += f"Total PnL: {total_pnl:.2f} USDT\n"
            report_content += f"Average PnL per Trade: {avg_pnl:.2f} USDT\n\n"
            
            report_content += "=== Trade Details ===\n\n"
            for trade in trades:
                status = "CLOSED" if trade['sell_time'] else "OPEN"
                pnl = f"{trade['profit_loss']:.2f} USDT" if trade['profit_loss'] is not None else "N/A"
                exit_price = f"{trade['sell_price']:.8f}" if trade['sell_price'] is not None else "N/A"
                
                report_content += f"Trade ID: {trade['id']}\n"
                report_content += f"Symbol: {trade['symbol']}\n"
                report_content += f"Entry: {trade['buy_price']:.8f}\n"
                report_content += f"Exit: {exit_price}\n"
                report_content += f"Status: {status}\n"
                report_content += f"PnL: {pnl}\n"
                report_content += "-------------------------\n"
        else:
            report_content = f"=== Daily Trading Report {today.strftime('%Y-%m-%d')} ===\n\n"
            report_content += f"Opening Capital: {opening_capital:.2f} USDT\n"
            report_content += "No trades executed today.\n"
        
        # Save report to file
        with open(report_filename, 'w') as f:
            f.write(report_content)
        
        print(f"\n[INFO] Daily report saved to {report_filename}")
        
    except Exception as e:
        print(f"[ERROR] Failed to generate daily report: {e}")

def display_trading_status(filtered_df, price_changes_map=None):
    """Display active trades and potential trading candidates"""
    print("\n=== Current Trading Status ===")
    
    # Display active trades
    print("\nActive Trades:")
    if active_trades:
        for symbol, trade in active_trades.items():
            current_price = None
            price_change = None
            if price_changes_map and symbol in price_changes_map:
                changes = price_changes_map[symbol]
                if 'live' in changes:
                    price_change = changes['live']
            
            if price_change:
                print(f"  {symbol}: Entry={trade.entry_price:.8f}, Live Change={price_change:+.2f}%")
            else:
                print(f"  {symbol}: Entry={trade.entry_price:.8f}")
    else:
        print("  No active trades")
    
    # Display potential candidates
    print("\nPotential Trades (Top 5 by Probability):")
    if not filtered_df.empty:
        top_candidates = filtered_df.head(5)
        for _, row in top_candidates.iterrows():
            symbol = row['symbol']
            prob = row['probability']
            changes = price_changes_map.get(symbol, {}) if price_changes_map else {}
            live_change = changes.get('live', 0)
            print(f"  {symbol}: Prob={prob:.2f}, Live Change={live_change:+.2f}%")
    else:
        print("  No potential trades found")
    print("============================\n")

def main():
    print("\n=== Trading Bot Started ===")
    print("Monitoring market for opportunities...")
    print("Press Ctrl+C to exit\n")
    
    # Load existing trades from database
    print("\n=== Loading Existing Trades ===")
    recovered_trades = fetch_active_trades_from_db()
    active_trades.update(recovered_trades)
    if recovered_trades:
        print(f"Recovered {len(recovered_trades)} active trades from database")
        for symbol, trade in recovered_trades.items():
            print(f"- {symbol}: Entry Price={trade.entry_price}, Quantity={trade.quantity}")
    print("============================\n")

    # Print spot wallet balance and performance stats
    print("\n=== Spot Wallet Balance ===")
    balances = get_spot_balance()
    if balances:
        for balance in balances:
            print(f"{balance['asset']}: Free={balance['free']:.8f}, Locked={balance['locked']:.8f}, Total={balance['total']:.8f}")
      # Calculate and display performance statistics
    stats = get_trading_stats(7)  # Get last 7 days stats
    if stats:
        print("\n=== Trading Statistics (7 Days) ===")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']}%")
        print(f"Total PnL: {stats['total_pnl']:.2f} USDT")
        print(f"Avg Profit/Trade: {stats['avg_profit_per_trade']:.2f} USDT")

        # Calculate and display compound growth projections
        growth_projections = calculate_compound_growth()
        if growth_projections:
            print("\n=== Compound Growth Projections ===")
            print(f"Average Profit per Trade: {growth_projections['avg_profit_per_trade_pct']}%")
            print("\nDaily Projection:")
            print(f"  Trades: {growth_projections['daily']['trades']}")
            print(f"  Profit %: {growth_projections['daily']['profit_pct']}%")
            print(f"  Growth: {growth_projections['daily']['projected_growth']:.2f} USDT")
            print("\nWeekly Projection:")
            print(f"  Trades: {growth_projections['weekly']['trades']}")
            print(f"  Profit %: {growth_projections['weekly']['profit_pct']}%")
            print(f"  Growth: {growth_projections['weekly']['projected_growth']:.2f} USDT")
            print("\nMonthly Projection:")
            print(f"  Trades: {growth_projections['monthly']['trades']}")
            print(f"  Profit %: {growth_projections['monthly']['profit_pct']}%")
            print(f"  Growth: {growth_projections['monthly']['projected_growth']:.2f} USDT")
            print("================================")

    # Update investment amount based on total capital
    global INVESTMENT_AMOUNT
    total_capital = get_total_capital()
    INVESTMENT_AMOUNT = get_dynamic_position_size(total_capital, "GENERAL")
    print(f"\nInitial Capital: {INITIAL_INVESTMENT:.2f} USDT")
    print(f"Current Total Capital: {total_capital:.2f} USDT")
    print(f"Current Position Size: {INVESTMENT_AMOUNT:.2f} USDT")
    print("=========================\n")
    
    while True:
        df = fetch_usdt_tickers()
        if df.empty:
            print("[INFO] No data returned from Binance API.")
            time.sleep(1)
            continue

        scored_df = calculate_score(df)        # Create an explicit copy of the DataFrame
        filtered_df = scored_df.head(10).copy()
        now = time.time()

        # Add probability column and sort
        filtered_df.loc[:, 'probability'] = filtered_df['symbol'].apply(
            lambda x: estimate_prob_3pct_jump_and_price(x)[0]
        )
        filtered_df = filtered_df.sort_values('probability', ascending=False)
        
        # Store price changes for all symbols to display later
        price_changes_map = {}
        
        for _, row in filtered_df.iterrows():
            symbol = row['symbol']
            if symbol in recent_symbols and (now - recent_symbols[symbol]) < SYMBOL_TTL:
                continue

            recent_symbols[symbol] = now
            price_changes = fetch_price_changes(symbol)
            prob_3pct, current_price, upper_price, lower_price = estimate_prob_3pct_jump_and_price(symbol)            # Store price changes for display
            price_changes_map[symbol] = price_changes

            if all(v == 0.0 for v in price_changes.values()):
                continue

            # Only process symbols with live price change greater than our target
            if price_changes['live'] < TARGET_PRICE_CHANGE:
                continue

            pnl = calculate_pnl(INVESTMENT_AMOUNT, price_changes['15m'], BINANCE_FEES_PCT)
            if pnl > 0:
                print(f"📈 {symbol}:")
                print(f"  24h: {price_changes['24h']:+.2f}% | 6h: {price_changes['6h']:+.2f}% | 1h: {price_changes['1h']:+.2f}%")
                print(f"  30m: {price_changes['30m']:+.2f}% | 15m: {price_changes['15m']:+.2f}% | 5m: {price_changes['5m']:+.2f}%")
                print(f"  Live: {price_changes['live']:+.2f}% | Prob≥3%: {prob_3pct}")
                print(f"  Vol={row['quoteVolume']:.2f}, PnL={pnl:.2f} USDT")
                print(f"  Price={current_price}, Range=({lower_price} - {upper_price})")

            # Check existing trades
            if symbol in active_trades:
                check_trade_exit(symbol, current_price)            # Evaluate new trading opportunities
            elif pnl > 0:
                evaluate_trading_opportunity(symbol, price_changes['15m'], price_changes['5m'], 
                                          prob_3pct, current_price, pnl)
                                          
        # Display current trading status after processing all symbols
        display_trading_status(filtered_df, price_changes_map)

        # Display active and potential trades status
        display_trading_status(filtered_df, price_changes_map={})

if __name__ == "__main__":
    # Initialize the recent_symbols dictionary
    recent_symbols = {}
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBot stopped by user.")
        generate_daily_report()  # Generate report when bot is stopped
    except Exception as e:
        print(f"\n\nBot stopped due to error: {e}")
        generate_daily_report()  # Generate report even if bot crashes
