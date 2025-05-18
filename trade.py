import os
import time
import pandas as pd
import requests
import hmac
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Dict, Optional, cast, Any
from dataclasses import dataclass
from config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Binance and Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
BINANCE_API_KEY = os.getenv("APIKey")
BINANCE_API_SECRET = os.getenv("secretKey")

if not all([SUPABASE_URL, SUPABASE_KEY, BINANCE_API_KEY, BINANCE_API_SECRET]):
    raise ValueError("Missing required environment variables")

# Cast environment variables to string since we've checked they exist
SUPABASE_URL = cast(str, SUPABASE_URL)
SUPABASE_KEY = cast(str, SUPABASE_KEY)
BINANCE_API_KEY = cast(str, BINANCE_API_KEY)
BINANCE_API_SECRET = cast(str, BINANCE_API_SECRET)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global state
active_trades = {}
recent_symbols = {}

# Initialize timeout from config
TRADE_TIMEOUT = timedelta(minutes=TRADE_TIMEOUT_MINUTES)

@dataclass
class SimulatedTrade:
    symbol: str
    entry_price: float
    quantity: float
    timestamp: datetime
    initial_pnl: float
    trade_id: Optional[int] = None

def fetch_price_data(symbol: str) -> Dict[str, Any]:
    """Fetch all necessary price data for a symbol in a single request"""
    try:
        # Get klines data for all timeframes in one request
        intervals = {
            '1d': ('1d', 2),
            '6h': ('6h', 2),
            '1h': ('1h', 2),
            '30m': ('30m', 2),
            '15m': ('15m', 2),
            '5m': ('5m', 2),
            '1m': ('1m', 1)  # Current candle only
        }
        
        price_data = {}
        for timeframe, (interval, limit) in intervals.items():
            endpoint = f"{BINANCE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()
            
            if timeframe == '1m' and len(data) >= 1:
                # For live price, compare open and current price of the current candle
                open_price = float(data[0][1])
                current_price = float(data[0][4])
                price_data[f'{timeframe}_change'] = round(((current_price - open_price) / open_price) * 100, 2)
                price_data['current_price'] = current_price
            elif len(data) >= 2:
                # For other timeframes, compare previous close with current close
                old_price = float(data[0][4])
                new_price = float(data[1][4])
                price_data[f'{timeframe}_change'] = round(((new_price - old_price) / old_price) * 100, 2)
        
        return price_data
    except Exception as e:
        logger.error(f"Failed to fetch price data for {symbol}: {e}")
        return {}

def fetch_usdt_tickers():
    """Fetch USDT trading pairs"""
    try:
        logger.info("Fetching USDT tickers...")
        response = requests.get(f"{BINANCE_URL}/ticker/24hr")
        response.raise_for_status()
        data = pd.DataFrame(response.json())
        usdt_tickers = data[data['symbol'].str.endswith("USDT")].copy()
        usdt_tickers['priceChangePercent'] = pd.to_numeric(usdt_tickers['priceChangePercent'], errors='coerce')
        usdt_tickers['quoteVolume'] = pd.to_numeric(usdt_tickers['quoteVolume'], errors='coerce')
        usdt_tickers['timestamp'] = datetime.now(timezone.utc).isoformat()
        return usdt_tickers[['symbol', 'priceChangePercent', 'quoteVolume', 'timestamp']]
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from Binance: {e}")
        return pd.DataFrame()

def calculate_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trading scores for symbols"""
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
    return df[['symbol', 'priceChangePercent', 'quoteVolume', 'composite_score', 'timestamp']].sort_values(
        by='composite_score', ascending=False)

def calculate_pnl(investment, price_change_pct, fees_pct):
    gross_return = investment * (price_change_pct / 100)
    trading_fees = investment * fees_pct * 2  # Entry and exit fees (0.075% * 2 = 0.15% total)
    trend_fee = investment * 0.001  # 0.1% trend fee
    net_return = gross_return - trading_fees - trend_fee
    return round(net_return, 2)

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
        logger.error(f"Failed to estimate 3% jump probability for {symbol}: {e}")
        return 0.0, 0.0, 0.0, 0.0

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
        logger.error(f"Failed to save trade entry to Supabase: {e}")
        return None

def update_trade_exit(trade_id: int, sell_price: float, profit_loss: float, exit_reason: str):
    """Update trade in Supabase with exit information"""
    try:
        data = {
            "sell_price": sell_price,
            "sell_time": datetime.now(timezone.utc).isoformat(),
            "profit_loss": profit_loss,
            "reason": exit_reason
        }
        supabase.table("trades").update(data).eq("id", trade_id).execute()
    except Exception as e:
        logger.error(f"Failed to update trade exit in Supabase: {e}")

def fetch_active_trades_from_db() -> Dict[str, SimulatedTrade]:
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
                initial_pnl=0.0,
                trade_id=trade['id']
            )
        return recovered_trades
    except Exception as e:
        logger.error(f"Failed to fetch active trades from database: {e}")
        return {}

def check_trade_exit(symbol: str, current_price: float) -> bool:
    """Check if we should exit a trade based on profit target, stop loss, or timeout"""
    if symbol not in active_trades:
        return False
        
    trade = active_trades[symbol]
    current_time = datetime.now(timezone.utc)
    
    if not current_price or current_price <= 0:
        logger.warning(f"Invalid exit price for {symbol}: {current_price}")
        return False
    
    price_change = ((current_price - trade.entry_price) / trade.entry_price) * 100
    
    timeout_exit = (current_time - trade.timestamp) > TRADE_TIMEOUT
    profit_target_reached = price_change >= TARGET_PRICE_CHANGE
    stop_loss_triggered = price_change <= STOP_LOSS_PCT
    
    if timeout_exit or profit_target_reached or stop_loss_triggered:
        if profit_target_reached:
            exit_reason = "Profit Target ✅"
        elif stop_loss_triggered:
            exit_reason = "Stop Loss ⛔"
        else:
            exit_reason = "Timeout ⏰"
        
        logger.info(f"\n[TRADE EXIT] {symbol}")
        logger.info(f"Entry Price: {trade.entry_price:.8f}")
        logger.info(f"Exit Price: {current_price:.8f} ({price_change:+.2f}%)")
        logger.info(f"Time in trade: {current_time - trade.timestamp}")
        logger.info(f"Investment: {INITIAL_INVESTMENT:.2f} USDT")
        logger.info(f"Reason: {exit_reason}")
        
        if trade.trade_id is not None:
            update_trade_exit(trade.trade_id, current_price, price_change, exit_reason)
        
        del active_trades[symbol]
        return True
    return False

def evaluate_trading_opportunity(symbol: str, price_data: Dict[str, Any]):
    """Evaluate if we should enter a trade based on price movements"""
    current_price = price_data.get('current_price')
    if not current_price or current_price <= 0:
        logger.warning(f"Invalid price for {symbol}: {current_price}")
        return
        
    # Check for positive price movements in recent timeframes
    one_min_change = price_data.get('1m_change', 0)
    five_min_change = price_data.get('5m_change', 0)
    fifteen_min_change = price_data.get('15m_change', 0)
    
    # Filter out if any recent timeframe had negative price change
    if one_min_change < 0 or five_min_change < 0 or fifteen_min_change < 0:
        logger.info(f"Skipping {symbol} due to negative recent price changes: "
                   f"1m: {one_min_change:+.2f}%, 5m: {five_min_change:+.2f}%, 15m: {fifteen_min_change:+.2f}%")
        return
        
    # Check for minimum price movement in last 5 minutes
    if five_min_change < MIN_PRICE_CHANGE:
        return

    # Skip if we already have maximum trades
    if len(active_trades) >= MAX_ACTIVE_TRADES:
        return
        
    # Check if trade exists in database but not in memory
    try:
        result = supabase.table("trades").select("*").eq("symbol", symbol).is_("sell_time", "null").execute()
        if result.data:
            trade_data = result.data[0]
            active_trades[symbol] = SimulatedTrade(
                symbol=symbol,
                entry_price=trade_data['buy_price'],
                quantity=trade_data['quantity'],
                timestamp=datetime.fromisoformat(trade_data['buy_time'].replace('Z', '+00:00')),
                initial_pnl=0.0,
                trade_id=trade_data['id']
            )
            logger.info(f"Recovered existing trade for {symbol} from database")
            return
    except Exception as e:
        logger.error(f"Failed to check existing trade in database: {e}")
        
    # Calculate quantity based on investment amount
    quantity = INITIAL_INVESTMENT / current_price
    
    # If criteria met, simulate trade entry
    if simulate_trade_entry(symbol, current_price, quantity, price_data.get('5m_change', 0)):
        logger.info(f"Started monitoring trade for {symbol}")

def simulate_trade_entry(symbol: str, current_price: float, quantity: float, initial_pnl: float) -> bool:
    """Simulate entering a trade"""
    if len(active_trades) >= MAX_ACTIVE_TRADES or symbol in active_trades:
        return False
    
    # Save trade to database first
    trade_id = save_trade_entry(symbol, current_price, quantity)
        
    trade = SimulatedTrade(
        symbol=symbol,
        entry_price=current_price,
        quantity=quantity,
        timestamp=datetime.now(timezone.utc),
        initial_pnl=initial_pnl,
        trade_id=trade_id
    )
    active_trades[symbol] = trade
    
    logger.info(f"[TRADE ENTRY] {symbol}")
    logger.info(f"Entry Price: {current_price:.8f}")
    logger.info(f"Quantity: {quantity:.4f}")
    logger.info(f"Initial PnL (5m change): {initial_pnl:.2f} USDT")
    logger.info(f"Trade ID: {trade_id}")
    logger.info("===============================")
    
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
            logger.error(f"Failed to fetch {tf} price change for {symbol}: {e}")
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
        logger.error(f"Failed to calculate total capital: {e}")
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
        logger.error(f"Failed to calculate trading stats: {e}")
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
        logger.error(f"Failed to calculate dynamic position size: {e}")
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
        logger.error(f"Failed to calculate compound growth: {e}")
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
        
        logger.info(f"Daily report saved to {report_filename}")
        
    except Exception as e:
        logger.error(f"Failed to generate daily report: {e}")

def display_trading_status(filtered_df: pd.DataFrame, price_data_map: Dict[str, Dict[str, Any]]):
    """Display active trades and potential trading candidates with detailed price information"""
    logger.info("\n" + "="*50)
    logger.info("TRADING STATUS REPORT")
    logger.info("="*50)
    
    # Display active trades with detailed price information
    logger.info("\n📊 ACTIVE TRADES:")
    if active_trades:
        for symbol, trade in active_trades.items():
            price_data = price_data_map.get(symbol, {})
            current_price = price_data.get('current_price', 0)
            if current_price:
                price_change = ((current_price - trade.entry_price) / trade.entry_price) * 100
                profit_loss = price_change > 0
                status_emoji = "📈" if profit_loss else "📉"
                
                logger.info(f"\n{status_emoji} {symbol}")
                logger.info(f"    Entry Price: {trade.entry_price:.8f}")
                logger.info(f"    Current Price: {current_price:.8f}")
                logger.info(f"    Change: {price_change:+.2f}%")
                logger.info(f"    Time in Trade: {datetime.now(timezone.utc) - trade.timestamp}")
                
                # Show price changes for different timeframes
                timeframes = ['1m', '5m', '15m', '1h', '6h']
                changes = [f"{tf}: {price_data.get(f'{tf}_change', 0):+.2f}%" for tf in timeframes]
                logger.info(f"    Price Changes: {' | '.join(changes)}")
    else:
        logger.info("  No active trades")
      # Display potential trading candidates
    logger.info("\n🎯 TOP TRADING CANDIDATES (Filtered for positive price action):")
    if not filtered_df.empty:
        logger.info(f"{'Symbol':<10} {'Price':<12} {'1m':<8} {'5m':<8} {'15m':<8} {'1h':<8} {'Vol(USDT)':<15} {'Status':<10}")
        logger.info("-"*85)
        
        for _, row in filtered_df.head(10).iterrows():
            symbol = row['symbol']
            price_data = price_data_map.get(symbol, {})
            current_price = price_data.get('current_price', 0)
            
            # Check if all recent timeframes are positive
            one_min = price_data.get('1m_change', 0)
            five_min = price_data.get('5m_change', 0)
            fifteen_min = price_data.get('15m_change', 0)
            one_hour = price_data.get('1h_change', 0)
              # Check for >3% price movement and positive timeframes
            status = "🔥 HOT" if one_min >= LIVE_PRICE_CHANGE_MIN else (
                     "✅ VALID" if all(x > 0 for x in [one_min, five_min, fifteen_min]) else "❌ FILTERED")
            
            if current_price:
                volume = row['quoteVolume'] / 1000000  # Convert to millions
                logger.info(
                    f"{symbol:<10} "
                    f"{current_price:<12.8f} "
                    f"{one_min:+6.2f}% "
                    f"{five_min:+6.2f}% "
                    f"{fifteen_min:+6.2f}% "
                    f"{one_hour:+6.2f}% "
                    f"{volume:>8.2f}M "
                    f"{status:<10}"
                )
    else:
        logger.info("  No potential trades found")
    
    # Market Summary
    logger.info("\n📈 MARKET SUMMARY:")
    if price_data_map:
        total_volume = sum(float(row['quoteVolume']) for _, row in filtered_df.iterrows()) / 1000000
        active_pairs = len(filtered_df)
        logger.info(f"Active USDT Pairs: {active_pairs}")
        logger.info(f"Total Volume: {total_volume:.2f}M USDT")
    
    logger.info("\n" + "="*50)

def main():
    logger.info("\n=== Trading Bot Started ===")
    logger.info("Monitoring market for opportunities...")
    
    # Load existing trades from database
    logger.info("\n=== Loading Existing Trades ===")
    recovered_trades = fetch_active_trades_from_db()
    active_trades.update(recovered_trades)
    if recovered_trades:
        logger.info(f"Recovered {len(recovered_trades)} active trades from database")
        for symbol, trade in recovered_trades.items():
            logger.info(f"- {symbol}: Entry Price={trade.entry_price}, Quantity={trade.quantity}")
    logger.info("============================\n")

    while True:
        try:
            df = fetch_usdt_tickers()
            if df.empty:
                logger.info("No data returned from Binance API.")
                time.sleep(1)
                continue

            scored_df = calculate_score(df)
            filtered_df = scored_df.head(10).copy()
            now = time.time()
            
            price_data_map = {}
            
            for _, row in filtered_df.iterrows():
                symbol = row['symbol']
                if symbol in recent_symbols and (now - recent_symbols[symbol]) < SYMBOL_TTL:
                    continue

                recent_symbols[symbol] = now
                price_data = fetch_price_data(symbol)
                price_data_map[symbol] = price_data

                if not price_data:
                    continue                # Only process symbols with live price change greater than 3%
                live_change = price_data.get('1m_change', 0)
                if live_change < LIVE_PRICE_CHANGE_MIN:
                    continue
                
                if live_change >= LIVE_PRICE_CHANGE_MIN:
                    logger.info(f"🔥 Strong momentum detected for {symbol}: {live_change:+.2f}% in 1 minute")

                current_price = price_data.get('current_price')
                
                # Check existing trades
                if current_price and symbol in active_trades:
                    check_trade_exit(symbol, float(current_price))
                elif current_price:
                    evaluate_trading_opportunity(symbol, price_data)
                                          
            # Display current trading status with detailed price information
            display_trading_status(filtered_df, price_data_map)
            
            # Add a small delay to prevent hitting rate limits
            time.sleep(2)

        except KeyboardInterrupt:
            logger.info("\n\nBot stopped by user.")
            generate_daily_report()
            break
        except Exception as e:
            logger.error(f"\n\nBot encountered an error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
