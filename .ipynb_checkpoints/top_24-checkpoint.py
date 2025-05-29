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

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class BinanceTopGainersBot:
    def __init__(self):
        # Use testnet URLs
        self.base_url = "https://testnet.binance.vision"
        self.ws_url = "wss://testnet.binance.vision/ws"
        self.price_changes = {}
        self.last_prices = {}
        self.max_tokens = 10  # Maximum number of tokens to trade
        self.target_net_profit = 0.02  # 0.02% target net profit
        self.maker_taker_fee = 0.1  # 0.1% fee per trade
        self.take_profit = self.target_net_profit + (2 * self.maker_taker_fee)  # 0.22% gross profit needed
        self.active_trades = {}  # Currently active trades
        self.top_gainers = []  # Store top gainers
        
        # Initialize Supabase client
        load_dotenv()
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logging.error("Supabase credentials not found in environment variables")
            raise ValueError("Supabase credentials not found in environment variables")
        
        logging.info("Initializing Supabase client...")
        self.supabase = create_client(supabase_url, supabase_key)
        logging.info("Supabase client initialized successfully")
        logging.info(f"Target net profit: {self.target_net_profit}%")
        logging.info(f"Trading fee per trade: {self.maker_taker_fee}%")
        logging.info(f"Required gross profit: {self.take_profit}%")

        # Initialize Binance Testnet API client
        self.binance_api_key = os.getenv("testAPI_Key")
        self.binance_api_secret = os.getenv("testAPI_Secret")
        if not self.binance_api_key or not self.binance_api_secret:
            logging.error("Binance testnet API credentials not found in environment variables")
            raise ValueError("Binance testnet API credentials not found in environment variables")
        
        # Initialize testnet client with base URLs
        self.binance_client = Client(
            api_key=self.binance_api_key,
            api_secret=self.binance_api_secret,
            tld='com'  # Use .com TLD for testnet
        )
        # Set testnet API URLs
        self.binance_client.API_URL = self.base_url
        self.binance_client.PRIVATE_API_VERSION = 'v3'
        logging.info("Binance Testnet API client initialized successfully")

        # Get USDT balance and set trading capital
        try:
            account_info = self.binance_client.get_account()
            usdt_balance = next(
                (float(balance['free']) for balance in account_info['balances'] if balance['asset'] == 'USDT'),
                0.0
            )
            self.trading_capital = usdt_balance * 0.9  # Use 90% of available USDT
            self.available_capital = self.trading_capital
            logging.info(f"Testnet USDT balance: {usdt_balance}")
            logging.info(f"Trading capital (90%): {self.trading_capital}")
        except Exception as e:
            logging.error(f"Error getting USDT balance: {e}")
            raise ValueError("Could not get USDT balance from Binance testnet")

    def calculate_fee(self, amount: float) -> float:
        """Calculate trading fee for a given amount."""
        return amount * (self.maker_taker_fee / 100)

    def calculate_quantity(self, price: float) -> float:
        """Calculate the quantity to buy based on available capital."""
        trade_amount = min(self.available_capital, 2.0)  # Use max 2 USDT per trade
        # Account for buy fee in quantity calculation
        fee = self.calculate_fee(trade_amount)
        adjusted_amount = trade_amount - fee
        quantity = Decimal(str(adjusted_amount / price)).quantize(Decimal('0.0000001'), rounding=ROUND_DOWN)
        return float(quantity)

    async def record_trade(self, trade_data: Dict):
        """Record trade in Supabase."""
        try:
            # Format the data according to the bot_trades table structure
            formatted_data = {
                'symbol': trade_data['symbol'],
                'buy_price': trade_data['buy_price'],
                'buy_time': trade_data['buy_time'],
                'sell_price': trade_data.get('sell_price'),
                'sell_time': trade_data.get('sell_time'),
                'quantity': trade_data['quantity'],
                'profit_loss': trade_data.get('profit_loss'),
                'reason': trade_data.get('reason')
            }
            
            # Insert into bot_trades table
            result = self.supabase.table('test_bot_trades').insert(formatted_data).execute()
            logging.info(f"Trade recorded successfully: {formatted_data}")
            
        except Exception as e:
            logging.error(f"Error recording trade: {e}")

    def get_top_gainers(self) -> List[str]:
        """Fetch top gainers in the last 24 hours from Binance."""
        try:
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
            logging.info(f"Top gainers: {', '.join(self.top_gainers)}")
            return self.top_gainers
            
        except Exception as e:
            logging.error(f"Error fetching top gainers: {e}")
            return []

    def get_symbol_info(self, symbol: str) -> dict | None:
        """Get symbol information with error handling and retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                symbol_info = self.binance_client.get_symbol_info(symbol)
                if symbol_info is None:
                    logging.error(f"Symbol {symbol} not found in exchange info")
                    return None
                return symbol_info
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to get symbol info for {symbol} after {max_retries} attempts: {e}")
                    return None
                logging.warning(f"Retry {attempt + 1}/{max_retries} getting symbol info for {symbol}")
                time.sleep(1)
        return None  # Return None if all retries failed

    async def handle_ticker_message(self, message: Dict):
        """Process incoming ticker websocket messages and handle trading."""
        symbol = message.get('s')
        current_price = float(message.get('c', 0))
        
        # Only process messages for top gainers
        if symbol not in self.top_gainers:
            return
        
        if symbol and current_price > 0:
            if symbol in self.last_prices:
                last_price = self.last_prices[symbol]
                price_change = ((current_price - last_price) / last_price) * 100
                
                # Handle active trades
                if symbol in self.active_trades:
                    trade = self.active_trades[symbol]
                    profit_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
                    
                    if profit_percent >= self.take_profit:
                        try:
                            # Execute sell order
                            sell_order = self.binance_client.create_order(
                                symbol=symbol,
                                side='SELL',
                                type='MARKET',
                                quantity=trade['quantity']
                            )
                            
                            if sell_order['status'] == 'FILLED':
                                # Calculate actual execution details
                                executed_qty = float(sell_order['executedQty'])
                                avg_price = float(sell_order['cummulativeQuoteQty']) / executed_qty
                                
                                # Calculate fees and profits
                                buy_fee = self.calculate_fee(trade['buy_price'] * trade['quantity'])
                                sell_fee = self.calculate_fee(avg_price * executed_qty)
                                total_fees = buy_fee + sell_fee
                                gross_profit = (avg_price - trade['buy_price']) * executed_qty
                                net_profit = gross_profit - total_fees
                                net_profit_percent = (net_profit / (trade['buy_price'] * trade['quantity'])) * 100
                                
                                await self.record_trade({
                                    'symbol': symbol,
                                    'buy_price': trade['buy_price'],
                                    'buy_time': trade['buy_time'].isoformat(),
                                    'sell_price': avg_price,
                                    'sell_time': datetime.now().isoformat(),
                                    'quantity': executed_qty,
                                    'profit_loss': net_profit,
                                    'reason': f'[TESTNET] Take profit hit - Gross: {profit_percent:.3f}%, Net: {net_profit_percent:.3f}%, Fees: {total_fees:.8f} USDT'
                                })
                                
                                self.available_capital += (avg_price * executed_qty) - sell_fee
                                del self.active_trades[symbol]
                                
                                logging.info(f"\n[TESTNET] Closed trade for {symbol}")
                                logging.info(f"Buy price: {trade['buy_price']:.8f} USDT")
                                logging.info(f"Sell price: {avg_price:.8f} USDT")
                                logging.info(f"Quantity: {executed_qty:.8f}")
                                logging.info(f"Gross profit: {profit_percent:.3f}%")
                                logging.info(f"Net profit: {net_profit_percent:.3f}%")
                                logging.info(f"Total fees: {total_fees:.8f} USDT")
                                
                        except Exception as e:
                            logging.error(f"[TESTNET] Error executing sell order for {symbol}: {str(e)}")
                
                # Check for new trade opportunities
                elif price_change > 0 and len(self.active_trades) < self.max_tokens:
                    quantity = self.calculate_quantity(current_price)
                    total_cost = (current_price * quantity)
                    
                    if total_cost <= self.available_capital and quantity > 0:
                        try:
                            # Get symbol info for precision
                            symbol_info = self.get_symbol_info(symbol)
                            if symbol_info is None:
                                logging.error(f"[TESTNET] Cannot trade {symbol}: Symbol information not available")
                                return
                                
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
                                    executed_qty = float(buy_order['executedQty'])
                                    avg_price = float(buy_order['cummulativeQuoteQty']) / executed_qty
                                    buy_time = datetime.now()
                                    buy_fee = self.calculate_fee(avg_price * executed_qty)
                                    
                                    await self.record_trade({
                                        'symbol': symbol,
                                        'buy_price': avg_price,
                                        'buy_time': buy_time.isoformat(),
                                        'quantity': executed_qty,
                                        'reason': f'[TESTNET] New position opened at {avg_price:.8f} USDT'
                                    })
                                    
                                    self.active_trades[symbol] = {
                                        'buy_price': avg_price,
                                        'quantity': executed_qty,
                                        'buy_time': buy_time,
                                        'buy_fee': buy_fee
                                    }
                                    self.available_capital -= (avg_price * executed_qty) + buy_fee
                                    
                                    logging.info(f"\n[TESTNET] Opening trade for {symbol}")
                                    logging.info(f"Buy price: {avg_price:.8f} USDT")
                                    logging.info(f"Quantity: {executed_qty:.8f}")
                                    logging.info(f"Buy fee: {buy_fee:.8f} USDT")
                                    logging.info(f"Total cost: {(avg_price * executed_qty + buy_fee):.8f} USDT")
                                    
                        except Exception as e:
                            logging.error(f"[TESTNET] Error executing buy order for {symbol}: {str(e)}")
            
            self.last_prices[symbol] = current_price

    async def display_status(self):
        """Display active trades and available capital."""
        while True:
            print("\033[2J\033[H")  # Clear screen
            print(f"Trading Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Available Capital: {self.available_capital:.2f} USDT")
            print("-" * 100)
            print(f"{'Symbol':<10} {'Price':<15} {'Buy Price':<15} {'Current P/L %':<12}")
            print("-" * 100)
            
            # Display active trades
            for symbol, trade in self.active_trades.items():
                current_price = self.last_prices.get(symbol, 0)
                if current_price > 0:
                    profit_percent = ((current_price - trade['buy_price']) / trade['buy_price']) * 100
                    print(
                        f"{symbol:<10} "
                        f"{current_price:<15.8f} "
                        f"{trade['buy_price']:<15.8f} "
                        f"{profit_percent:>10.2f}%"
                    )
            
            await asyncio.sleep(1)

    async def update_top_gainers(self):
        """Periodically update the list of top gainers."""
        while True:
            self.get_top_gainers()
            await asyncio.sleep(300)  # Update every 5 minutes

    async def connect_websocket(self):
        """Connect to Binance WebSocket and process messages."""
        try:
            # Get initial top gainers
            symbols = self.get_top_gainers()
            logging.info(f"Monitoring {len(symbols)} trading pairs...")
            
            # Create a subscription payload instead of URL parameters
            subscribe_payload = {
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@ticker" for symbol in symbols],
                "id": 1
            }
            
            logging.info("Connecting to WebSocket...")
            
            while True:
                try:
                    async with websockets.client.connect(self.ws_url) as websocket:
                        logging.info("WebSocket connected successfully!")
                        
                        # Send subscription request
                        await websocket.send(json.dumps(subscribe_payload))
                        logging.info("Subscription request sent")
                        
                        while True:
                            try:
                                message = await websocket.recv()
                                data = json.loads(message)
                                
                                # Skip subscription confirmation messages
                                if 'result' in data:
                                    continue
                                    
                                await self.handle_ticker_message(data)
                            except websockets.exceptions.ConnectionClosed as e:
                                logging.error(f"WebSocket connection closed unexpectedly: {e}")
                                raise
                            except json.JSONDecodeError as e:
                                logging.error(f"Failed to decode message: {e}")
                                continue
                            except Exception as e:
                                logging.error(f"Error processing message: {e}")
                                continue
                                
                except websockets.exceptions.InvalidStatusCode as e:
                    logging.error(f"Invalid status code from server: {e}")
                    if e.status_code == 404:
                        logging.error("WebSocket endpoint not found (404). Please check if the endpoint URL is correct.")
                    raise
                except websockets.exceptions.InvalidURI as e:
                    logging.error(f"Invalid WebSocket URI: {e}")
                    raise
                except Exception as e:
                    logging.error(f"WebSocket connection error: {e}")
                    raise
                
        except Exception as e:
            logging.error(f"Fatal error in connect_websocket: {e}")
            print(f"WebSocket error: {e}")
            print("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
            return await self.connect_websocket()

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
            logging.error(f"Error fetching spot wallet balances: {e}")
            return {}

    async def display_spot_wallet_balances(self):
        """Display spot wallet balances periodically."""
        while True:
            balances = self.get_spot_wallet_balances()
            print("\nSpot Wallet Balances:")
            for asset, amount in balances.items():
                print(f"{asset}: {amount}")
            # Print USDT spot balance after line 336
            usdt_balance = balances.get('USDT', 0)
            print(f"\nYour USDT Spot Balance: {usdt_balance}")
            await asyncio.sleep(10)

    async def main(self):
        """Main function to run the bot."""
        print("Starting top gainers trading bot...")
        print(f"Initial capital: {self.trading_capital} USDT")
        print(f"Take profit: {self.take_profit}%")
        
        tasks = [
            asyncio.create_task(self.connect_websocket()),
            asyncio.create_task(self.display_status()),
            asyncio.create_task(self.update_top_gainers()),
            asyncio.create_task(self.display_spot_wallet_balances())
        ]
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    bot = BinanceTopGainersBot()
    asyncio.run(bot.main())
