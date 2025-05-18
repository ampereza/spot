import asyncio
from binance.client import AsyncClient
from binance import BinanceSocketManager

async def process_message(msg):
    if msg.get('e') == 'trade':
        symbol = msg['s']
        price = msg['p']
        print(f"{symbol}: {price}")

async def main():
    client = await AsyncClient.create()
    
    # Get all USDT trading pairs
    prices = await client.get_all_tickers()
    usdt_pairs = [p['symbol'] for p in prices if p['symbol'].endswith('USDT')]

    # Initialize BinanceSocketManager
    bsm = BinanceSocketManager(client)

    # Create trade streams for a few USDT pairs
    selected_pairs = usdt_pairs[:5]  # limit to 5 pairs
    stream_names = [f"{symbol.lower()}@trade" for symbol in selected_pairs]

    # Start multiplex WebSocket
    ts = bsm.multiplex_socket(stream_names)
    async with ts as tscm:
        while True:
            msg = await tscm.recv()
            await process_message(msg)

    await client.close_connection()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped by user.")
