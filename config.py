"""Trading bot configuration settings"""

# Initial investment and trade settings
INITIAL_INVESTMENT = 100
MAX_ACTIVE_TRADES = 3
HIGH_PROB_THRESHOLD = 0.5

# Price movement thresholds
TARGET_PRICE_CHANGE = 0.35  # Target 0.35% price change for taking profits
LIVE_PRICE_CHANGE_MIN = 3.0  # Minimum live price change to consider entry (3%)
STOP_LOSS_PCT = -1.0  # Stop loss at 10% loss
MIN_PRICE_CHANGE = 0.05  # Minimum price movement to consider entry

# Trading fees and parameters
BINANCE_FEES_PCT = 0.00075  # 0.075% when paying with BNB

# Time settings
TRADE_TIMEOUT_MINUTES = 300
SYMBOL_TTL = 5  # Time to live for symbol cache in seconds

# API Settings
BINANCE_URL = "https://api.binance.com/api/v3"
ENDPOINTS = {
    'ticker': '/ticker/24hr',
    'klines': '/klines',
    'account': '/account'
}

# Investment sizing
POSITION_SIZE_MULTIPLIER = 0.95  # Use 95% of available capital for position sizing
CONSERVATIVE_POSITION_SIZE = 0.60  # Conservative position size multiplier
MODERATE_POSITION_SIZE = 0.80  # Moderate position size multiplier
AGGRESSIVE_POSITION_SIZE = 0.95  # Aggressive position size multiplier
