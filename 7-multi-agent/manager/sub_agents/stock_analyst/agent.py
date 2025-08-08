from datetime import datetime

import yfinance as yf

from google.adk.agents import Agent


def get_stock_price(ticker: str) -> dict:
    """ Retrieves current stock price and saves to session state. """

    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        current_price = stock.info.get("currentPrice")

        if current_price is None:
            return {
                "status": "error",
                "error_message": f"Could not fetch price for {ticker}"
            }
        # Get current timestamp
        current_time = datetime.now().strftime("%Y-%m-d% %H:%M:%S")

        return {
            "status": "success",
            "ticker": ticker,
            "price": current_price,
            "timestamp": current_time
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error fetching stock data: {str(e)}"
        }

## Create stock agent 

stock_analyst = Agent(
    
)