from binance.client import Client
import time
import requests




def get_client(api_key, api_secret):
    client = Client(api_key, api_secret)
    client.API_URL = 'https://testnet.binance.vision/api'
    return client


def get_price(client, symbol):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None
    

def buy_crypto(client, symbol, quantity):
    try:
        order = client.order_market_buy(
            symbol=symbol,
            quantity=quantity
        )
        return order
    except Exception as e:
        print(f"Error placing buy order for {symbol}: {e}")
        return None
    

def sell_crypto(client, symbol, quantity):
    try:
        order = client.order_market_sell(
            symbol=symbol,
            quantity=quantity
        )
        return order
    except Exception as e:
        print(f"Error placing sell order for {symbol}: {e}")
        return None
    
def get_top_gainers_24h(client, top_n=10):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)
        response.raise_for_status()
        tickers = response.json()
        # Filter only symbols ending with USDT for simplicity
        usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
        # Sort by percent change in descending order
        sorted_tickers = sorted(usdt_tickers, key=lambda x: float(x['priceChangePercent']), reverse=True)
        return sorted_tickers[:top_n]
    except Exception as e:
        print(f"Error fetching top gainers: {e}")
        return []

def get_top_gainers_1min(client, top_n=5):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)
        response.raise_for_status()
        tickers = response.json()
        usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
        # For 1min gainers, fetch 1min klines and calculate percent change
        gainers = []
        for t in usdt_tickers:
            symbol = t['symbol']
            try:
                klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=2)
                if len(klines) < 2:
                    continue
                open_price = float(klines[0][1])
                close_price = float(klines[1][4])
                percent_change = ((close_price - open_price) / open_price) * 100
                gainers.append({'symbol': symbol, 'percent_change': percent_change})
            except Exception:
                continue
        sorted_gainers = sorted(gainers, key=lambda x: x['percent_change'], reverse=True)
        return sorted_gainers[:top_n]
    except Exception as e:
        print(f"Error fetching 1min top gainers: {e}")
        return []

def place_limit_buy_order(client, symbol, price, quantity):
    try:
        order = client.order_limit_buy(
            symbol=symbol,
            price=str(price),
            quantity=quantity
        )
        return order
    except Exception as e:
        print(f"Error placing limit buy order for {symbol}: {e}")
        return None

def place_limit_sell_order(client, symbol, price, quantity):
    try:
        order = client.order_limit_sell(
            symbol=symbol,
            price=str(price),
            quantity=quantity
        )
        return order
    except Exception as e:
        print(f"Error placing limit sell order for {symbol}: {e}")
        return None
    
def trade_top_gainers(client, top_n=5, usdt_amount=10):
    gainers = get_top_gainers_24h(client, top_n=top_n)
    print(f"\nPlacing limit buy and sell orders for the top {top_n} gainers against USDT:")
    for t in gainers:
        symbol = t['symbol']
        try:
            price = get_price(client, symbol)
            if price is None or price <= 0:
                print(f"Skipping {symbol}: could not fetch price.")
                continue
            buy_price = round(price * 0.9995, 2)  # 0.05% less than current price, rounded to 2 decimals
            sell_price = round(price * 1.003, 2)  # 0.3% more than current price, rounded to 2 decimals
            quantity = usdt_amount / buy_price
            quantity = round(quantity, 6)
            buy_order = place_limit_buy_order(client, symbol, buy_price, quantity)
            if buy_order:
                print(f"Limit buy order placed for {symbol}: {quantity} at {buy_price} USDT.")
                # Place sell order only if buy order is placed
                sell_order = place_limit_sell_order(client, symbol, sell_price, quantity)
                if sell_order:
                    print(f"Limit sell order placed for {symbol}: {quantity} at {sell_price} USDT.")
                else:
                    print(f"Failed to place limit sell order for {symbol}.")
            else:
                print(f"Failed to place limit buy order for {symbol}.")
        except Exception as e:
            print(f"Error trading {symbol}: {e}")

# Main function to continuously fetch and print the price of the cryptocurrency
    
def main(api_key, api_secret, symbol):
    client = get_client(api_key, api_secret)
    top_gainers_1min = get_top_gainers_1min(client)
    print("Top 5 gainers in the last 1 minute:")
    for t in top_gainers_1min:
        print(f"{t['symbol']}: {t['percent_change']:.2f}%")
    top_gainers = get_top_gainers_24h(client)
    print("Top 10 gainers in the last 24 hours:")
    for t in top_gainers:
        print(f"{t['symbol']}: {t['priceChangePercent']}%")
    trade_top_gainers(client, top_n=5, usdt_amount=10)
    while True:
        price = get_price(client, symbol)
        if price is not None:
            print(f"The current price of {symbol} is: {price}")
        else:
            print("Failed to retrieve price.")
        time.sleep(5)  # Wait for 5 seconds before the next request


if __name__ == "__main__":
    testAPI_Key="OdjvnSKFy7xBtFwcjF6aqyumuXdiKwZYh1KbeB799HrmuNne6A8BZcIp90GUTcCq"
    testAPI_Secret="oOho1J9cLOOwL1dziUDr3Afcjz8TzKZY4DBxpLEtIx7vu57DXvgWW9K9rgtZK2JI"
    symbol = "BTCUSDT"
    main(testAPI_Key, testAPI_Secret, symbol)
