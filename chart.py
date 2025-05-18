from binance.client import Client
from binance.streams import BinanceSocketManager
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from collections import defaultdict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PriceChart:
    def __init__(self):
        self.client = Client(
            os.getenv("APIKey"),
            os.getenv("secretKey")
        )
        self.timeframes = {
            '1m': {'interval': Client.KLINE_INTERVAL_1MINUTE, 'limit': 60},
            '5m': {'interval': Client.KLINE_INTERVAL_5MINUTE, 'limit': 60},
            '15m': {'interval': Client.KLINE_INTERVAL_15MINUTE, 'limit': 60},
            '30m': {'interval': Client.KLINE_INTERVAL_30MINUTE, 'limit': 48},
            '6h': {'interval': Client.KLINE_INTERVAL_6HOUR, 'limit': 40},
            '1d': {'interval': Client.KLINE_INTERVAL_1DAY, 'limit': 30}
        }
        self.candles = defaultdict(dict)

    def fetch_historical_candles(self, symbol: str):
        for tf, params in self.timeframes.items():
            klines = self.client.get_klines(
                symbol=symbol,
                interval=params['interval'],
                limit=params['limit']
            )
            self.candles[tf][symbol] = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ]).astype({
                'open': float, 'high': float, 'low': float,
                'close': float, 'volume': float
            })

    def plot_charts(self, symbol: str):
        self.fetch_historical_candles(symbol)
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('24h', '6h', '30m', '15m', '5m', '1m'),
            vertical_spacing=0.05
        )

        for i, (tf, params) in enumerate(self.timeframes.items()):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            df = self.candles[tf][symbol]
            
            fig.add_trace(
                go.Candlestick(
                    x=pd.to_datetime(df['timestamp'], unit='ms'),
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=tf
                ),
                row=row, col=col
            )

        fig.update_layout(
            title=f'{symbol} Price Charts',
            height=1000,
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            xaxis3_rangeslider_visible=False,
            xaxis4_rangeslider_visible=False,
            xaxis5_rangeslider_visible=False,
            xaxis6_rangeslider_visible=False
        )

        fig.show()

if __name__ == "__main__":
    chart = PriceChart()
    symbol = "BTCUSDT"  # Example symbol
    chart.plot_charts(symbol)
