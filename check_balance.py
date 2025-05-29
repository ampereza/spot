from binance.client import Client
from dotenv import load_dotenv
import os

def check_spot_wallet():
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from environment variables
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("Error: Binance API credentials not found in environment variables")
        print("Please make sure you have BINANCE_API_KEY and BINANCE_API_SECRET in your .env file")
        return
    
    try:
        # Initialize Binance client
        client = Client(api_key, api_secret)
        
        # Get account information
        account_info = client.get_account()
        balances = account_info.get('balances', [])
        
        # Filter and display non-zero balances
        print("\nYour Binance Spot Wallet Balances:")
        print("-" * 50)
        print(f"{'Asset':<10} {'Free':<15} {'Locked':<15}")
        print("-" * 50)
        
        for balance in balances:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                print(f"{balance['asset']:<10} {free:<15.8f} {locked:<15.8f}")
                
    except Exception as e:
        print(f"\nError connecting to Binance: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify your API keys are correct")
        print("3. Make sure you're not behind a restrictive firewall")
        print("4. If using a VPN, try disabling it temporarily")

if __name__ == "__main__":
    check_spot_wallet() 