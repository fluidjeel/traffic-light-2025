import yfinance as yf
import pandas as pd

# Stocks already available in FYERS (no need to check)
FYERS_AVAILABLE = {
    "IDEA", "YESBANK", "PNBHOUSING", "CENTRALBK", "UCOBANK", 
    "IRB", "ADANIPOWER", "GLENMARK", "AJANTPHARM", "JUBLFOOD",
    "PAGEIND", "PETRONET", "SUZLON"
}

# Remaining delisted stocks to check in yfinance
REMAINING_DELISTED = {
    "DHFL.NS": "Dewan Housing",
    "RCOM.NS": "Reliance Communications",
    "RPOWER.NS": "Reliance Power",
    "RELINFRA.NS": "Reliance Infra",
    "JPASSOCIAT.NS": "Jaiprakash Associates",
    "JETAIRWAYS.NS": "Jet Airways",
    "FRETAIL.NS": "Future Retail",
    "ALOKTEXT.NS": "Alok Industries",
    "UNITECH.NS": "Unitech",
    "BALLARPUR.NS": "Ballarpur Ind.",
    "SINTEX.NS": "Sintex",
    "LANCOTECH.NS": "Lanco Infra",
    "ESSARSTEEL.NS": "Essar Steel",
    "IBULHSGFIN.NS": "Indiabulls Housing"
}

def check_yfinance_availability():
    results = []
    
    for symbol, name in REMAINING_DELISTED.items():
        try:
            data = yf.Ticker(symbol).history(period="5y")
            available = not data.empty
            start_date = data.index[0].strftime("%Y-%m-%d") if available else "N/A"
            results.append({
                "Symbol": symbol.replace(".NS", ""),
                "Company": name,
                "Available": available,
                "Start_Date": start_date,
                "Rows": len(data) if available else 0
            })
            print(f"{symbol}: {'✅' if available else '❌'}")
        except Exception as e:
            print(f"⚠️ Error checking {symbol}: {str(e)}")
            results.append({
                "Symbol": symbol.replace(".NS", ""),
                "Company": name,
                "Available": False,
                "Start_Date": "Error",
                "Rows": 0
            })
    
    pd.DataFrame(results).to_csv("yfinance_delisted_status.csv", index=False)

if __name__ == "__main__":
    check_yfinance_availability()