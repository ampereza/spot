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

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class BinanceTopGainersBot:
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.price_changes = {}
        self.last_prices = {}
        self.trading_capital = 20.0  # Total USDT for trading
        self.max_tokens = 10  # Maximum number of tokens to trade
        self.target_net_profit = 0.02  # 0.02% target net profit
        self.maker_taker_fee = 0.1  # 0.1% fee per trade
        self.take_profit = self.target_net_profit + (2 * self.maker_taker_fee)  # 0.22% gross profit needed
        self.active_trades = {}  # Currently active trades
        self.available_capital = self.trading_capital
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
            result = self.supabase.table('bot_trades').insert(formatted_data).execute()
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
                        # Calculate fees and profits
                        buy_fee = self.calculate_fee(trade['buy_price'] * trade['quantity'])
                        sell_fee = self.calculate_fee(current_price * trade['quantity'])
                        total_fees = buy_fee + sell_fee
                        gross_profit = (current_price - trade['buy_price']) * trade['quantity']
                        net_profit = gross_profit - total_fees
                        net_profit_percent = (net_profit / (trade['buy_price'] * trade['quantity'])) * 100
                        
                        await self.record_trade({
                            'symbol': symbol,
                            'buy_price': trade['buy_price'],
                            'buy_time': trade['buy_time'].isoformat(),
                            'sell_price': current_price,
                            'sell_time': datetime.now().isoformat(),
                            'quantity': trade['quantity'],
                            'profit_loss': net_profit,
                            'reason': f'Take profit hit - Gross: {profit_percent:.3f}%, Net: {net_profit_percent:.3f}%, Fees: {total_fees:.8f} USDT'
                        })
                        
                        self.available_capital += (current_price * trade['quantity']) - sell_fee
                        del self.active_trades[symbol]
                        print(f"\nClosed trade for {symbol}")
                        print(f"Buy price: {trade['buy_price']:.8f} USDT")
                        print(f"Sell price: {current_price:.8f} USDT")
                        print(f"Quantity: {trade['quantity']:.8f}")
                        print(f"Gross profit: {profit_percent:.3f}%")
                        print(f"Net profit: {net_profit_percent:.3f}%")
                        print(f"Total fees: {total_fees:.8f} USDT")
                
                # Check for new trade opportunities
                elif price_change > 0 and len(self.active_trades) < self.max_tokens:
                    quantity = self.calculate_quantity(current_price)
                    buy_fee = self.calculate_fee(current_price * quantity)
                    total_cost = (current_price * quantity) + buy_fee
                    
                    if total_cost <= self.available_capital and quantity > 0:
                        buy_time = datetime.now()
                        target_sell = current_price * (1 + (self.take_profit / 100))
                        
                        await self.record_trade({
                            'symbol': symbol,
                            'buy_price': current_price,
                            'buy_time': buy_time.isoformat(),
                            'quantity': quantity,
                            'reason': f'New position opened at {current_price:.8f} USDT (Target: {target_sell:.8f} USDT, Fee: {buy_fee:.8f} USDT)'
                        })
                        
                        self.active_trades[symbol] = {
                            'buy_price': current_price,
                            'quantity': quantity,
                            'buy_time': buy_time,
                            'buy_fee': buy_fee
                        }
                        self.available_capital -= total_cost
                        
                        print(f"\nOpening trade for {symbol}")
                        print(f"Buy price: {current_price:.8f} USDT")
                        print(f"Target sell: {target_sell:.8f} USDT")
                        print(f"Quantity: {quantity:.8f}")
                        print(f"Buy fee: {buy_fee:.8f} USDT")
                        print(f"Total cost: {total_cost:.8f} USDT")
            
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

    async def main(self):
        """Main function to run the bot."""
        print("Starting top gainers trading bot...")
        print(f"Initial capital: {self.trading_capital} USDT")
        print(f"Take profit: {self.take_profit}%")
        
        tasks = [
            asyncio.create_task(self.connect_websocket()),
            asyncio.create_task(self.display_status()),
            asyncio.create_task(self.update_top_gainers())
        ]
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    bot = BinanceTopGainersBot()
    asyncio.run(bot.main())
