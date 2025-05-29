from binance.client import Client
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def test_binance_connection():
    """Test Binance testnet connection and basic functionality."""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("testAPI_Key")
        api_secret = os.getenv("testAPI_Secret")

        if not api_key or not api_secret:
            raise ValueError("API credentials not found in .env file")

        # Initialize testnet client
        client = Client(api_key=api_key, api_secret=api_secret)
        client.API_URL = 'https://testnet.binance.vision/api'

        # Test 1: Ping the server
        logging.info("Test 1: Pinging Binance testnet...")
        client.ping()
        logging.info("✓ Successfully connected to Binance testnet")

        # Test 2: Get server time
        logging.info("\nTest 2: Checking server time...")
        server_time = client.get_server_time()
        server_datetime = datetime.fromtimestamp(server_time['serverTime'] / 1000)
        logging.info(f"✓ Server time: {server_datetime}")

        # Test 3: Get account information
        logging.info("\nTest 3: Getting account information...")
        account_info = client.get_account()
        balances = {
            asset['asset']: {
                'free': float(asset['free']),
                'locked': float(asset['locked'])
            }
            for asset in account_info['balances']
            if float(asset['free']) > 0 or float(asset['locked']) > 0
        }
        logging.info("✓ Account balances:")
        for asset, amounts in balances.items():
            logging.info(f"  {asset}: Free={amounts['free']}, Locked={amounts['locked']}")

        # Test 4: Get BTCUSDT price
        logging.info("\nTest 4: Getting BTCUSDT price...")
        btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
        logging.info(f"✓ Current BTCUSDT price: {btc_price['price']}")

        # Test 5: Get trading pair information
        logging.info("\nTest 5: Getting trading pair information...")
        trading_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        for pair in trading_pairs:
            try:
                symbol_info = client.get_symbol_info(pair)
                if symbol_info:
                    filters = {f['filterType']: f for f in symbol_info['filters']}
                    logging.info(f"\n✓ {pair} trading rules:")
                    if 'LOT_SIZE' in filters:
                        logging.info(f"  Minimum quantity: {filters['LOT_SIZE']['minQty']}")
                        logging.info(f"  Maximum quantity: {filters['LOT_SIZE']['maxQty']}")
                        logging.info(f"  Step size: {filters['LOT_SIZE']['stepSize']}")
                    if 'PRICE_FILTER' in filters:
                        logging.info(f"  Minimum price: {filters['PRICE_FILTER']['minPrice']}")
                        logging.info(f"  Maximum price: {filters['PRICE_FILTER']['maxPrice']}")
            except Exception as e:
                logging.error(f"Error getting info for {pair}: {str(e)}")

        # Test 6: Test market data endpoints
        logging.info("\nTest 6: Testing market data endpoints...")
        try:
            # Get recent trades
            trades = client.get_recent_trades(symbol='BTCUSDT', limit=1)
            logging.info("✓ Recent trades endpoint working")
            
            # Get historical klines
            klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "5 minutes ago UTC")
            logging.info("✓ Historical klines endpoint working")
            
            # Get 24hr ticker
            ticker_24hr = client.get_ticker(symbol='BTCUSDT')
            logging.info("✓ 24hr ticker endpoint working")
            
        except Exception as e:
            logging.error(f"Error testing market data endpoints: {str(e)}")

        logging.info("\n✅ All basic functionality tests completed successfully!")
        return True

    except Exception as e:
        logging.error(f"\n❌ Error during testing: {str(e)}")
        logging.error("\nTroubleshooting tips:")
        logging.error("1. Check if your API keys are correct in the .env file")
        logging.error("2. Ensure you have internet connectivity")
        logging.error("3. Check if Binance testnet is accessible")
        logging.error("4. Verify your API keys have the necessary permissions")
        return False

if __name__ == "__main__":
    test_binance_connection() 