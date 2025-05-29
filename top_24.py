import asyncio
import json
import websockets.client
import websockets.exceptions
from datetime import datetime
from typing import Dict, List
from supabase.client import create_client
from dotenv import load_dotenv
import os
from decimal import Decimal, ROUND_DOWN
import urllib.request
import urllib.parse
import logging
from binance.client import Client
import time
import colorama
from colorama import Fore, Style
import sys

# Initialize colorama with Windows support
colorama.init(wrap=True)

# Configure logging with Windows-compatible output
class WindowsCompatibleFormatter(logging.Formatter):
    """Custom formatter that works with Windows console"""
    
    def format(self, record):
        # Store the original message
        original_msg = record.msg
        
        # Add color based on level
        if record.levelno == logging.INFO:
            record.msg = f"{original_msg}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{Fore.YELLOW}{original_msg}{Style.RESET_ALL}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{Fore.RED}{original_msg}{Style.RESET_ALL}"
            
        return super().format(record)

# Configure logging
logger = logging.getLogger('BinanceBot')
logger.setLevel(logging.INFO)

# Console handler with Windows-compatible formatter
console_handler = logging.StreamHandler(sys.stdout)  # Explicitly use stdout
console_handler.setLevel(logging.INFO)
console_formatter = WindowsCompatibleFormatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler for keeping logs (without colors)
file_handler = logging.FileHandler('bot_trading.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

class BinanceTopGainersBot:
    def __init__(self):
        logger.info("Initializing Binance Top Gainers Bot...")
        
        # Use testnet URLs
        self.base_url = "https://testnet.binance.vision"
        self.ws_url = "wss://stream.binance.com:9443/ws"  # Using main Binance WebSocket for price feeds
        self.price_changes = {}
        self.last_prices = {}
        self.price_history = {}  # Store historical prices for different timeframes
        self.timeframes = {
            '6h': 21600,    # 6 hours in seconds
            '1h': 3600,     # 1 hour
            '30m': 1800,    # 30 minutes
            '5m': 300,      # 5 minutes
            '1m': 60,       # 1 minute
            'live': 0       # Current price change
        }
        self.max_tokens = 10
        self.target_net_profit = 0.02
        self.maker_taker_fee = 0.1
        self.take_profit = self.target_net_profit + (2 * self.maker_taker_fee)
        self.active_trades = {}
        self.top_gainers = []
        self.stop_loss = 0.01  # 1% stop loss
        self.ws_connected = False  # Track WebSocket connection state
        
        # Initialize Supabase client
        load_dotenv()
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found in environment variables")
            raise ValueError("Supabase credentials not found in environment variables")
        
        logger.info("Initializing Supabase client...")
        self.supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully")
        
        # Log trading parameters
        logger.info(f"Trading Parameters:")
        logger.info(f"├── Max Tokens: {self.max_tokens}")
        logger.info(f"├── Target Net Profit: {self.target_net_profit}%")
        logger.info(f"├── Trading Fee: {self.maker_taker_fee}%")
        logger.info(f"└── Take Profit: {self.take_profit}%")

        # Initialize Binance Testnet API client
        self.binance_api_key = os.getenv("testAPI_Key")
        self.binance_api_secret = os.getenv("testAPI_Secret")
        if not self.binance_api_key or not self.binance_api_secret:
            logger.error("Binance testnet API credentials not found in environment variables")
            raise ValueError("Binance testnet API credentials not found in environment variables")
        
        try:
            # Initialize testnet client
            self.binance_client = Client(
                api_key=self.binance_api_key,
                api_secret=self.binance_api_secret
            )
            
            # Configure for testnet
            self.binance_client.API_URL = 'https://testnet.binance.vision/api'
            self.binance_client.PRIVATE_API_VERSION = 'v3'
            
            # Test API connection
            self.binance_client.ping()
            logger.info("Successfully connected to Binance testnet")
            
            # Get account information
            account_info = self.binance_client.get_account()
            usdt_balance = next(
                (float(balance['free']) for balance in account_info['balances'] if balance['asset'] == 'USDT'),
                0.0
            )
            self.trading_capital = usdt_balance * 0.9
            self.per_trade_amount = self.trading_capital / self.max_tokens
            self.available_capital = self.trading_capital
            
            logger.info("Account Information:")
            logger.info(f"├── Total USDT Balance: {usdt_balance:.2f}")
            logger.info(f"├── Trading Capital (90%): {self.trading_capital:.2f}")
            logger.info(f"└── Amount per Trade: {self.per_trade_amount:.2f}")
            
        except Exception as e:
            logger.error(f"Error connecting to Binance testnet: {str(e)}")
            raise ValueError("Could not connect to Binance testnet")

    def calculate_fee(self, amount: float) -> float:
        """Calculate trading fee for a given amount."""
        return amount * (self.maker_taker_fee / 100)

    def calculate_quantity(self, price: float) -> float:
        """Calculate the quantity to buy based on per trade amount."""
        # Use the per-trade amount for each trade
        trade_amount = self.per_trade_amount
        
        # Account for buy fee in quantity calculation
        fee = self.calculate_fee(trade_amount)
        adjusted_amount = trade_amount - fee
        
        # Calculate quantity with precision
        quantity = Decimal(str(adjusted_amount / price)).quantize(Decimal('0.0000001'), rounding=ROUND_DOWN)
        return float(quantity)

    async def record_trade(self, trade_data: Dict):
        """Record trade in Supabase test_bot_trades table."""
        try:
            formatted_data = {
                'symbol': trade_data['symbol'],
                'buy_price': float(trade_data['buy_price']),
                'buy_time': trade_data['buy_time'],
                'sell_price': float(trade_data['sell_price']) if 'sell_price' in trade_data else None,
                'sell_time': trade_data['sell_time'] if 'sell_time' in trade_data else None,
                'quantity': float(trade_data['quantity']),
                'profit_loss': float(trade_data['profit_loss']) if 'profit_loss' in trade_data else None,
                'reason': trade_data['reason']
            }
            
            result = self.supabase.table('test_bot_trades').insert(formatted_data).execute()
            logger.info(f"Trade recorded in database:")
            logger.info(f"├── Symbol: {formatted_data['symbol']}")
            logger.info(f"├── Buy Price: {formatted_data['buy_price']}")
            logger.info(f"├── Quantity: {formatted_data['quantity']}")
            if formatted_data['sell_price']:
                logger.info(f"├── Sell Price: {formatted_data['sell_price']}")
                logger.info(f"└── Profit/Loss: {formatted_data['profit_loss']}")
            else:
                logger.info(f"└── Status: Position Opened")
            
        except Exception as e:
            logger.error(f"Error recording trade in test_bot_trades: {e}")

    def get_top_gainers(self) -> List[str]:
        """Fetch top gainers in the last 24 hours from Binance."""
        try:
            logger.info("Fetching top gainers from Binance...")
            url = f"{self.base_url}/api/v3/ticker/24hr"
            response = urllib.request.urlopen(url)
            data = json.loads(response.read())
            
            # Filter USDT pairs and calculate price change
            usdt_pairs = [
                {
                    'symbol': item['symbol'],
                    'priceChange': float(item['priceChangePercent'])
                }
                for item in data
                if item['symbol'].endswith('USDT')
            ]
            
            # Sort by price change percentage (descending)
            sorted_pairs = sorted(
                usdt_pairs,
                key=lambda x: x['priceChange'],
                reverse=True
            )
            
            # Get top 20 gainers
            self.top_gainers = [pair['symbol'] for pair in sorted_pairs[:20]]
            
            logger.info("Top Gainers Updated:")
            for idx, symbol in enumerate(self.top_gainers[:10], 1):
                price_change = next(p['priceChange'] for p in sorted_pairs if p['symbol'] == symbol)
                logger.info(f"├── {idx}. {symbol}: {price_change:+.2f}%")
            logger.info("└── ...")
            
            return self.top_gainers
            
        except Exception as e:
            logger.error(f"Error fetching top gainers: {e}")
            return []

    def get_symbol_info(self, symbol: str) -> dict | None:
        """Get symbol information with error handling and retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                symbol_info = self.binance_client.get_symbol_info(symbol)
                if symbol_info is None:
                    logger.error(f"Symbol {symbol} not found in exchange info")
                    return None
                return symbol_info
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to get symbol info for {symbol} after {max_retries} attempts: {e}")
                    return None
                logger.warning(f"Retry {attempt + 1}/{max_retries} getting symbol info for {symbol}")
                time.sleep(1)
        return None  # Return None if all retries failed

    async def execute_buy_trade(self, symbol: str, current_price: float) -> bool:
        """Execute a buy trade for the given symbol at the current price."""
        try:
            quantity = self.calculate_quantity(current_price)
            total_cost = (current_price * quantity)
            
            if total_cost <= self.available_capital and quantity > 0:
                # Get symbol info for precision
                symbol_info = self.get_symbol_info(symbol)
                if symbol_info is None:
                    logger.error(f"[TESTNET] Cannot trade {symbol}: Symbol information not available")
                    return False
                    
                lot_size_filter = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', symbol_info['filters']))
                step_size = float(lot_size_filter['stepSize'])
                
                # Adjust quantity to match lot size
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                adjusted_quantity = float(Decimal(str(quantity)).quantize(Decimal(str(step_size)), rounding=ROUND_DOWN))
                
                if adjusted_quantity > 0:
                    # Execute buy order
                    buy_order = self.binance_client.create_order(
                        symbol=symbol,
                        side='BUY',
                        type='MARKET',
                        quantity=adjusted_quantity
                    )
                    
                    if buy_order['status'] == 'FILLED':
                        await self.process_filled_buy_order(symbol, buy_order)
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"[TESTNET] Error executing buy order for {symbol}: {str(e)}")
            return False

    async def process_filled_buy_order(self, symbol: str, buy_order: dict):
        """Process a filled buy order and update relevant data structures."""
        executed_qty = float(buy_order['executedQty'])
        avg_price = float(buy_order['cummulativeQuoteQty']) / executed_qty
        buy_fee = self.calculate_fee(avg_price * executed_qty)
        buy_time = datetime.now()
        
        # Record new trade with buy details
        await self.record_trade({
            'symbol': symbol,
            'buy_price': avg_price,
            'buy_time': buy_time.isoformat(),
            'quantity': executed_qty,
            'reason': f'New position opened at {avg_price:.8f} USDT'
        })
        
        self.active_trades[symbol] = {
            'buy_price': avg_price,
            'quantity': executed_qty,
            'buy_time': buy_time,
            'buy_fee': buy_fee
        }
        self.available_capital -= (avg_price * executed_qty) + buy_fee
        
        logger.info(f"\n[TESTNET] Opening trade for {symbol}")
        logger.info(f"Buy price: {avg_price:.8f} USDT")
        logger.info(f"Quantity: {executed_qty:.8f}")
        logger.info(f"Buy fee: {buy_fee:.8f} USDT")
        logger.info(f"Total cost: {(avg_price * executed_qty + buy_fee):.8f} USDT")

    async def execute_sell_trade(self, symbol: str, trade: dict, current_price: float, reason: str) -> bool:
        """Execute a sell trade for the given symbol."""
        try:
            sell_order = self.binance_client.create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=trade['quantity']
            )
            
            if sell_order['status'] == 'FILLED':
                await self.process_filled_sell_order(symbol, sell_order, trade, reason)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"[TESTNET] Error executing sell order for {symbol}: {str(e)}")
            return False

    async def process_filled_sell_order(self, symbol: str, sell_order: dict, trade: dict, reason: str):
        """Process a filled sell order and update relevant data structures."""
        executed_qty = float(sell_order['executedQty'])
        avg_price = float(sell_order['cummulativeQuoteQty']) / executed_qty
        
        # Calculate fees and profits
        buy_fee = trade.get('buy_fee', 0)
        sell_fee = self.calculate_fee(avg_price * executed_qty)
        total_fees = buy_fee + sell_fee
        gross_profit = (avg_price - trade['buy_price']) * executed_qty
        net_profit = gross_profit - total_fees
        net_profit_percent = (net_profit / (trade['buy_price'] * trade['quantity'])) * 100
        profit_percent = ((avg_price - trade['buy_price']) / trade['buy_price']) * 100
        
        # Record completed trade with sell details
        await self.record_trade({
            'symbol': symbol,
            'buy_price': trade['buy_price'],
            'buy_time': trade['buy_time'].isoformat(),
            'sell_price': avg_price,
            'sell_time': datetime.now().isoformat(),
            'quantity': executed_qty,
            'profit_loss': net_profit,
            'reason': f'{reason} - Gross: {profit_percent:.3f}%, Net: {net_profit_percent:.3f}%, Fees: {total_fees:.8f} USDT'
        })
        
        self.available_capital += (avg_price * executed_qty) - sell_fee
        del self.active_trades[symbol]
        
        logger.info(f"\n[TESTNET] Closed trade for {symbol}")
        logger.info(f"Buy price: {trade['buy_price']:.8f} USDT")
        logger.info(f"Sell price: {avg_price:.8f} USDT")
        logger.info(f"Quantity: {executed_qty:.8f}")
        logger.info(f"Gross profit: {profit_percent:.3f}%")
        logger.info(f"Net profit: {net_profit_percent:.3f}%")
        logger.info(f"Total fees: {total_fees:.8f} USDT")

    def calculate_price_change(self, symbol: str, timeframe: str) -> float:
        """Calculate price change for a specific timeframe."""
        try:
            if symbol not in self.price_history:
                return 0.0
            
            history = self.price_history[symbol].get(timeframe, [])
            if not history:
                return 0.0
            
            current_price = history[-1]['price']
            
            if timeframe == 'live':
                if len(history) >= 2:
                    previous_price = history[-2]['price']
                    return ((current_price - previous_price) / previous_price) * 100
                return 0.0
            
            # For other timeframes, find the oldest price within the timeframe
            oldest_price = None
            cutoff_time = datetime.now().timestamp() - self.timeframes[timeframe]
            
            for price_data in history:
                if price_data['timestamp'] <= cutoff_time:
                    oldest_price = price_data['price']
                    break
            
            if oldest_price is None and len(history) > 1:
                oldest_price = history[0]['price']
            elif oldest_price is None:
                return 0.0
            
            return ((current_price - oldest_price) / oldest_price) * 100
            
        except Exception as e:
            logger.error(f"Error calculating price change for {symbol} {timeframe}: {e}")
            return 0.0

    async def update_price_history(self, symbol: str, price: float):
        """Update price history for a symbol."""
        try:
            current_time = datetime.now().timestamp()
            
            # Initialize price history for this symbol if it doesn't exist
            if symbol not in self.price_history:
                self.price_history[symbol] = {}
            
            # Update live price history
            if 'live' not in self.price_history[symbol]:
                self.price_history[symbol]['live'] = []
            
            self.price_history[symbol]['live'].append({
                'timestamp': current_time,
                'price': price
            })
            
            # Keep only the last 2 prices for live updates
            if len(self.price_history[symbol]['live']) > 2:
                self.price_history[symbol]['live'] = self.price_history[symbol]['live'][-2:]
            
            # Update other timeframes
            for timeframe in ['6h', '1h', '30m', '5m', '1m']:
                if timeframe not in self.price_history[symbol]:
                    self.price_history[symbol][timeframe] = []
                
                history = self.price_history[symbol][timeframe]
                
                # Add new price point if enough time has passed
                if not history or (current_time - history[-1]['timestamp']) >= self.timeframes[timeframe]:
                    history.append({
                        'timestamp': current_time,
                        'price': price
                    })
                else:
                    # Update the latest price
                    history[-1]['price'] = price
                
                # Keep only relevant historical data
                cutoff_time = current_time - self.timeframes[timeframe]
                self.price_history[symbol][timeframe] = [
                    data for data in history
                    if data['timestamp'] > cutoff_time
                ]
            
        except Exception as e:
            logger.error(f"Error updating price history for {symbol}: {e}")

    async def handle_ticker_message(self, message: Dict):
        """Process incoming ticker websocket messages and handle trading."""
        symbol = message.get('s')
        current_price = float(message.get('c', 0))
        
        # Only process messages for top gainers
        if symbol not in self.top_gainers:
            return
        
        if symbol and current_price > 0:
            # Update price history before processing
            await self.update_price_history(symbol, current_price)
            
            if symbol in self.last_prices:
                last_price = self.last_prices[symbol]
                price_change = ((current_price - last_price) / last_price) * 100
                
                # Handle active trades
                if symbol in self.active_trades:
                    trade = self.active_trades[symbol]
                    profit_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
                    
                    if profit_percent >= self.take_profit:
                        await self.execute_sell_trade(symbol, trade, current_price, 'Take profit hit')
                
                # Check for new trade opportunities
                elif price_change > 0 and len(self.active_trades) < self.max_tokens:
                    await self.execute_buy_trade(symbol, current_price)
            
            self.last_prices[symbol] = current_price

    async def get_trade_history(self):
        """Fetch trade history from Supabase."""
        try:
            response = self.supabase.table('test_bot_trades')\
                .select('*')\
                .order('buy_time', desc=True)\
                .limit(10)\
                .execute()
            
            return response.data
        except Exception as e:
            logger.error(f"Error fetching trade history: {e}")
            return []

    async def display_trade_history(self):
        """Display the last 10 completed trades."""
        trades = await self.get_trade_history()
        
        if trades:
            logger.info("\nRecent Trades:")
            logger.info("─" * 100)
            logger.info(f"{'Symbol':<10} {'Buy Price':<12} {'Sell Price':<12} {'Quantity':<12} {'P/L %':<10} {'P/L USDT':<12} {'Time':<20}")
            logger.info("─" * 100)
            
            for trade in trades:
                if trade.get('sell_price'):  # Only show completed trades
                    symbol = trade['symbol']
                    buy_price = float(trade['buy_price'])
                    sell_price = float(trade['sell_price'])
                    quantity = float(trade['quantity'])
                    profit_loss = float(trade['profit_loss'])
                    profit_percent = (profit_loss / (buy_price * quantity)) * 100
                    
                    # Format the time to be more readable
                    trade_time = datetime.fromisoformat(trade['sell_time'].replace('Z', '+00:00'))
                    formatted_time = trade_time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Color code based on profit/loss
                    color = Fore.YELLOW
                    if profit_loss > 0:
                        color = Fore.GREEN
                    elif profit_loss < 0:
                        color = Fore.RED
                    
                    logger.info(
                        f"{symbol:<10} "
                        f"{buy_price:<12.8f} "
                        f"{sell_price:<12.8f} "
                        f"{quantity:<12.8f} "
                        f"{color}{profit_percent:>+9.2f}%{Style.RESET_ALL} "
                        f"{color}{profit_loss:>+11.2f}{Style.RESET_ALL} "
                        f"{formatted_time:<20}"
                    )
            
            logger.info("─" * 100)
        else:
            logger.info("\nNo completed trades yet.")

    async def display_status(self):
        """Display active trades and available capital."""
        update_interval = 5  # Update every 5 seconds
        clear_counter = 0
        max_clear_count = 12  # Clear screen every 60 seconds (12 * 5)
        trade_history_counter = 0
        trade_history_interval = 6  # Show trade history every 30 seconds (6 * 5)
        
        while True:
            try:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Only clear screen periodically to make it more readable
                clear_counter += 1
                if clear_counter >= max_clear_count:
                    print("\033[2J\033[H")  # Clear screen
                    clear_counter = 0
                    logger.info("Press Ctrl+C to exit the bot")
                
                # Display header
                logger.info(f"\n=== Trading Status - {current_time} ===")
                logger.info(f"Capital Status:")
                logger.info(f"├── USDT Available: {self.available_capital:.2f}")
                logger.info(f"├── Trade Size: {self.per_trade_amount:.2f} USDT")
                logger.info(f"└── Active Trades: {len(self.active_trades)}/{self.max_tokens}")
                
                # Display top gainers being monitored in table format
                logger.info("\nTop Trading Pairs:")
                logger.info("─" * 160)  # Increased width for more columns
                logger.info(
                    f"{'Symbol':<10} {'Price':<15} {'6h':<8} {'1h':<8} {'30m':<8} "
                    f"{'5m':<8} {'1m':<8} {'Live':<8} {'Status':<10}"
                )
                logger.info("─" * 160)
                
                for symbol in self.top_gainers[:10]:
                    current_price = self.last_prices.get(symbol, 0)
                    
                    # Get price changes for all timeframes
                    changes = {
                        tf: self.calculate_price_change(symbol, tf)
                        for tf in self.timeframes.keys()
                    }
                    
                    # Determine status
                    status = "Monitoring"
                    if symbol in self.active_trades:
                        status = "Trading"
                    
                    # Format each timeframe with color
                    timeframe_formats = []
                    for tf in ['6h', '1h', '30m', '5m', '1m', 'live']:
                        change = changes[tf]
                        color = Fore.YELLOW
                        if change > 0:
                            color = Fore.GREEN
                        elif change < 0:
                            color = Fore.RED
                        timeframe_formats.append(f"{color}{change:>+6.2f}%{Style.RESET_ALL}")
                    
                    logger.info(
                        f"{symbol:<10} "
                        f"{current_price:<15.8f} "
                        f"{timeframe_formats[0]:<8} "
                        f"{timeframe_formats[1]:<8} "
                        f"{timeframe_formats[2]:<8} "
                        f"{timeframe_formats[3]:<8} "
                        f"{timeframe_formats[4]:<8} "
                        f"{timeframe_formats[5]:<8} "
                        f"{status:<10}"
                    )
                
                logger.info("─" * 160)
                
                # Display active trades
                if self.active_trades:
                    logger.info("\nActive Trades:")
                    logger.info("─" * 100)
                    logger.info(f"{'Symbol':<10} {'Current':<15} {'Entry':<15} {'P/L %':<12} {'Duration':<10}")
                    logger.info("─" * 100)
                    
                    for symbol, trade in self.active_trades.items():
                        current_price = self.last_prices.get(symbol, 0)
                        if current_price > 0:
                            profit_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
                            duration = datetime.now() - trade['buy_time']
                            hours = duration.total_seconds() / 3600
                            
                            color = Fore.YELLOW
                            if profit_percent >= self.take_profit:
                                color = Fore.GREEN
                            elif profit_percent < 0:
                                color = Fore.RED
                            
                            logger.info(
                                f"{symbol:<10} "
                                f"{current_price:<15.8f} "
                                f"{trade['buy_price']:<15.8f} "
                                f"{color}{profit_percent:>10.2f}%{Style.RESET_ALL} "
                                f"{hours:>9.1f}h"
                            )
                    
                    logger.info("─" * 100)
                
                # Display trade history periodically
                trade_history_counter += 1
                if trade_history_counter >= trade_history_interval:
                    await self.display_trade_history()
                    trade_history_counter = 0
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in display_status: {e}")
                await asyncio.sleep(update_interval)

    async def connect_websocket(self):
        """Connect to Binance WebSocket and process messages."""
        retry_delay = 1
        max_retry_delay = 60
        self.ws_connected = False
        
        while True:
            try:
                # Get initial top gainers
                symbols = self.get_top_gainers()
                if not symbols:
                    logger.error("No symbols available for monitoring. Retrying...")
                    await asyncio.sleep(retry_delay)
                    continue

                logger.info(f"Monitoring {len(symbols)} trading pairs...")
                
                # Create a subscription payload
                subscribe_payload = {
                    "method": "SUBSCRIBE",
                    "params": [
                        f"{symbol.lower()}@miniTicker" for symbol in symbols
                    ],
                    "id": 1
                }
                
                logger.info(f"Connecting to WebSocket at {self.ws_url}...")
                
                try:
                    async with websockets.client.connect(
                        self.ws_url,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=10,
                        max_size=2**20,  # 1MB max message size
                        extra_headers={
                            'User-Agent': 'Mozilla/5.0',
                        },
                        compression=None  # Disable compression to reduce complexity
                    ) as websocket:
                        logger.info("WebSocket connected successfully!")
                        self.ws_connected = True
                        retry_delay = 1  # Reset retry delay on successful connection
                        
                        try:
                            # Send subscription request with timeout
                            subscription_timeout = 5  # 5 seconds timeout for subscription
                            await asyncio.wait_for(
                                websocket.send(json.dumps(subscribe_payload)),
                                timeout=subscription_timeout
                            )
                            
                            # Wait for subscription confirmation with timeout
                            confirmation = await asyncio.wait_for(
                                websocket.recv(),
                                timeout=subscription_timeout
                            )
                            conf_data = json.loads(confirmation)
                            if conf_data.get('result') is None:
                                logger.warning("No subscription confirmation received")
                            else:
                                logger.info("Subscription confirmed")
                            
                            # Initialize prices using REST API
                            await self.initialize_prices(symbols)
                            
                            while True:
                                try:
                                    # Use timeout for receiving messages
                                    message = await asyncio.wait_for(
                                        websocket.recv(),
                                        timeout=30  # 30 seconds timeout
                                    )
                                    data = json.loads(message)
                                    
                                    # Skip subscription confirmation messages
                                    if 'result' in data:
                                        continue
                                        
                                    # Update price data from miniTicker
                                    if 's' in data:  # Symbol exists in data
                                        symbol = data['s']
                                        if symbol in self.top_gainers:  # Only process if in our monitored list
                                            try:
                                                current_price = float(data['c'])
                                                if current_price > 0:  # Validate price
                                                    self.last_prices[symbol] = current_price
                                                else:
                                                    logger.warning(f"Received invalid price for {symbol}: {current_price}")
                                            except (ValueError, KeyError) as e:
                                                logger.error(f"Error processing price for {symbol}: {e}")
                                                continue
                                    
                                    await self.handle_ticker_message(data)
                                    
                                except asyncio.TimeoutError:
                                    logger.warning("WebSocket receive timeout, checking connection...")
                                    # Send a ping to check if connection is still alive
                                    pong_waiter = await websocket.ping()
                                    try:
                                        await asyncio.wait_for(pong_waiter, timeout=5)
                                    except asyncio.TimeoutError:
                                        logger.error("WebSocket ping timeout, reconnecting...")
                                        break
                                except websockets.exceptions.ConnectionClosed as e:
                                    logger.warning(f"WebSocket connection closed: {str(e)}")
                                    break
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to decode message: {e}")
                                    continue
                                except Exception as e:
                                    logger.error(f"Error processing message: {str(e)}")
                                    continue
                                    
                        except asyncio.TimeoutError:
                            logger.error("Timeout during WebSocket operation")
                            continue
                        except Exception as e:
                            logger.error(f"Error in WebSocket message loop: {str(e)}")
                            continue
                        finally:
                            self.ws_connected = False
                            
                except websockets.exceptions.InvalidStatusCode as e:
                    logger.error(f"Invalid status code from WebSocket server: {e.status_code}")
                    await asyncio.sleep(retry_delay)
                except websockets.exceptions.InvalidMessage as e:
                    logger.error(f"Invalid WebSocket message: {str(e)}")
                    await asyncio.sleep(retry_delay)
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"WebSocket connection closed unexpectedly: {str(e)}")
                    await asyncio.sleep(retry_delay)
                except Exception as e:
                    logger.error(f"WebSocket connection error: {str(e)}")
                    await asyncio.sleep(retry_delay)
                    
            except Exception as e:
                logger.error(f"Main WebSocket loop error: {str(e)}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
                continue
            
            # If we get here, try to reconnect after a delay
            logger.info(f"Attempting to reconnect in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)

    async def initialize_prices(self, symbols: List[str]):
        """Initialize prices using REST API."""
        try:
            timeframe_intervals = {
                '6h': '1h',
                '1h': '1m',
                '30m': '1m',
                '5m': '1m',
                '1m': '1m'
            }
            
            for symbol in symbols:
                try:
                    # Initialize price history structure
                    if symbol not in self.price_history:
                        self.price_history[symbol] = {}
                    
                    # Get current price
                    ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    self.last_prices[symbol] = current_price
                    
                    # Initialize 'live' timeframe
                    self.price_history[symbol]['live'] = [{
                        'timestamp': datetime.now().timestamp(),
                        'price': current_price
                    }]
                    
                    # Get historical klines for each timeframe
                    for timeframe, interval in timeframe_intervals.items():
                        try:
                            # Calculate number of candles needed
                            if timeframe == '6h':
                                limit = 6  # 6 hours using 1h candles
                            elif timeframe == '1h':
                                limit = 60  # 60 minutes
                            elif timeframe == '30m':
                                limit = 30  # 30 minutes
                            elif timeframe == '5m':
                                limit = 5   # 5 minutes
                            else:  # 1m
                                limit = 1   # 1 minute
                            
                            klines = self.binance_client.get_klines(
                                symbol=symbol,
                                interval=interval,
                                limit=limit
                            )
                            
                            if klines:
                                self.price_history[symbol][timeframe] = [
                                    {
                                        'timestamp': k[0] / 1000,  # Convert ms to seconds
                                        'price': float(k[4])  # Close price
                                    }
                                    for k in klines
                                ]
                                
                                # Add current price as the latest point
                                self.price_history[symbol][timeframe].append({
                                    'timestamp': datetime.now().timestamp(),
                                    'price': current_price
                                })
                        
                        except Exception as e:
                            logger.error(f"Error getting {timeframe} data for {symbol}: {e}")
                            self.price_history[symbol][timeframe] = []
                    
                    # Get 24h stats
                    ticker_24h = self.binance_client.get_ticker(symbol=symbol)
                    self.price_changes[symbol] = float(ticker_24h['priceChangePercent'])
                    
                except Exception as e:
                    logger.error(f"Error initializing price for {symbol}: {e}")
                
                await asyncio.sleep(0.1)  # Small delay to avoid rate limits
                
        except Exception as e:
            logger.error(f"Error in initialize_prices: {e}")

    async def update_top_gainers(self):
        """Periodically update the list of top gainers."""
        update_interval = 60  # Update every 60 seconds
        while True:
            try:
                symbols = self.get_top_gainers()
                
                # Update prices and changes
                for symbol in symbols:
                    try:
                        # Get 24h ticker for price change
                        ticker_24h = self.binance_client.get_ticker(symbol=symbol)
                        self.price_changes[symbol] = float(ticker_24h['priceChangePercent'])
                        
                        # Update current price if not available from WebSocket
                        if symbol not in self.last_prices or self.last_prices[symbol] == 0:
                            self.last_prices[symbol] = float(ticker_24h['lastPrice'])
                    except Exception as e:
                        logger.error(f"Error updating {symbol}: {e}")
                    await asyncio.sleep(0.1)  # Small delay to avoid rate limits
                
                await asyncio.sleep(update_interval)
            except Exception as e:
                logger.error(f"Error in update_top_gainers: {e}")
                await asyncio.sleep(update_interval)

    def get_spot_wallet_balances(self):
        """Fetch spot wallet balances from Binance."""
        try:
            # Get account information
            account_info = self.binance_client.get_account()
            balances = account_info.get('balances', [])
            
            # Filter and format balances
            formatted_balances = {
                balance['asset']: float(balance['free']) + float(balance['locked'])
                for balance in balances
                if float(balance['free']) + float(balance['locked']) > 0
            }
            
            return formatted_balances
            
        except Exception as e:
            logger.error(f"Error fetching spot wallet balances: {e}")
            return {}

    async def display_spot_wallet_balances(self):
        """Display USDT balance periodically."""
        update_interval = 30  # Update every 30 seconds
        while True:
            try:
                balances = self.get_spot_wallet_balances()
                usdt_balance = balances.get('USDT', 0)
                logger.info(f"\nWallet Balance:")
                logger.info(f"└── USDT: {usdt_balance:.2f}")
                await asyncio.sleep(update_interval)
            except Exception as e:
                logger.error(f"Error in display_spot_wallet_balances: {e}")
                await asyncio.sleep(update_interval)

    async def log_order_details(self, order: Dict, order_type: str, symbol: str, reason: str = ''):
        """Log order details to Supabase database."""
        try:
            executed_qty = float(order['executedQty'])
            avg_price = float(order['cummulativeQuoteQty']) / executed_qty if executed_qty > 0 else 0
            
            order_data = {
                'order_id': str(order['orderId']),
                'symbol': symbol,
                'order_type': order_type,
                'side': order['side'],
                'status': order['status'],
                'price': avg_price,
                'quantity': executed_qty,
                'total_quote': float(order['cummulativeQuoteQty']),
                'commission': self.calculate_fee(avg_price * executed_qty),
                'timestamp': datetime.now().isoformat(),
                'reason': reason or 'No reason provided'  # Ensure reason is always a string
            }
            
            # Insert into orders table
            result = self.supabase.table('test_orders').insert(order_data).execute()
            logger.info(f"Order logged successfully: {order_data}")
            return order_data
            
        except Exception as e:
            logger.error(f"Error logging order details: {e}")
            return None

    async def main(self):
        """Main function to run the bot."""
        logger.info("\n=== Starting Binance Top Gainers Trading Bot ===")
        logger.info(f"Initial capital: {self.trading_capital:.2f} USDT")
        logger.info(f"Take profit target: {self.take_profit:.2f}%")
        logger.info("Press Ctrl+C to exit the bot\n")
        
        tasks = []
        try:
            tasks = [
                asyncio.create_task(self.connect_websocket()),
                asyncio.create_task(self.display_status()),
                asyncio.create_task(self.update_top_gainers()),
                asyncio.create_task(self.display_spot_wallet_balances())
            ]
            
            # Wait for tasks to complete or KeyboardInterrupt
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            logger.info("\nReceived shutdown signal. Gracefully shutting down the bot...")
            
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            
        finally:
            # Cancel all tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    
            # Wait for all tasks to complete their cleanup
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            
            # Final cleanup
            if self.ws_connected:
                logger.info("Closing WebSocket connection...")
                self.ws_connected = False
            
            logger.info("Bot shutdown complete. Goodbye!")

if __name__ == "__main__":
    bot = BinanceTopGainersBot()
    asyncio.run(bot.main())