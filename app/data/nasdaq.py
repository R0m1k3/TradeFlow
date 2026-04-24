"""
TradeFlow — Stock List with Company Names
Provides ticker → (company name, currency) mapping.
Fetches from yfinance, falls back to a curated list.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Curated list: ticker → (company name, currency)
# Covers NASDAQ + major European exchanges (Euronext Paris, etc.)
STOCK_INFO: dict[str, tuple[str, str]] = {
    # ── Mega cap NASDAQ ──
    "AAPL":   ("Apple", "USD"),
    "MSFT":   ("Microsoft", "USD"),
    "NVDA":   ("NVIDIA", "USD"),
    "AMZN":   ("Amazon", "USD"),
    "GOOG":   ("Alphabet", "USD"),
    "GOOGL":  ("Alphabet", "USD"),
    "META":   ("Meta Platforms", "USD"),
    "TSLA":   ("Tesla", "USD"),
    # ── Large cap NASDAQ ──
    "AVGO":   ("Broadcom", "USD"),
    "COST":   ("Costco", "USD"),
    "NFLX":   ("Netflix", "USD"),
    "AMD":    ("AMD", "USD"),
    "CRM":    ("Salesforce", "USD"),
    "ADBE":   ("Adobe", "USD"),
    "PEP":    ("PepsiCo", "USD"),
    "CSCO":   ("Cisco", "USD"),
    "INTC":   ("Intel", "USD"),
    "CMCSA":  ("Comcast", "USD"),
    "TXN":    ("Texas Instruments", "USD"),
    "QCOM":   ("Qualcomm", "USD"),
    "SBUX":   ("Starbucks", "USD"),
    "AMGN":   ("Amgen", "USD"),
    "VRTX":   ("Vertex Pharma", "USD"),
    "BKNG":   ("Booking Holdings", "USD"),
    "ISRG":   ("Intuitive Surgical", "USD"),
    "REGN":   ("Regeneron", "USD"),
    "FISV":   ("Fiserv", "USD"),
    "GILD":   ("Gilead Sciences", "USD"),
    "ADP":    ("ADP", "USD"),
    "CSX":    ("CSX", "USD"),
    "MU":     ("Micron", "USD"),
    "LRCX":   ("Lam Research", "USD"),
    "PANW":   ("Palo Alto Networks", "USD"),
    "KLAC":   ("KLA Corporation", "USD"),
    "SNPS":   ("Synopsys", "USD"),
    "MCHP":   ("Microchip Tech", "USD"),
    "CDW":    ("CDW", "USD"),
    "MAR":    ("Marriott", "USD"),
    "MRVL":   ("Marvell Tech", "USD"),
    "ORLY":   ("O'Reilly Auto", "USD"),
    "FTNT":   ("Fortinet", "USD"),
    "MNST":   ("Monster Beverage", "USD"),
    "DLTR":   ("Dollar Tree", "USD"),
    "CPRT":   ("Copart", "USD"),
    "CTAS":   ("Cintas", "USD"),
    "FAST":   ("Fastenal", "USD"),
    "IDXX":   ("IDEXX Labs", "USD"),
    "BIIB":   ("Biogen", "USD"),
    "VRSK":   ("Verisk Analytics", "USD"),
    "ILMN":   ("Illumina", "USD"),
    "CEG":    ("Constellation Energy", "USD"),
    "WDAY":   ("Workday", "USD"),
    "XEL":    ("Xcel Energy", "USD"),
    "EXC":    ("Exelon", "USD"),
    "WBA":    ("Walgreens", "USD"),
    "MDLZ":   ("Mondelez", "USD"),
    "KDP":    ("Keurig Dr Pepper", "USD"),
    "ROST":   ("Ross Stores", "USD"),
    "CHTR":   ("Charter Comm.", "USD"),
    "PYPL":   ("PayPal", "USD"),
    # ── Tech growth ──
    "ABNB":   ("Airbnb", "USD"),
    "ZS":     ("Zscaler", "USD"),
    "DDOG":   ("Datadog", "USD"),
    "CRWD":   ("CrowdStrike", "USD"),
    "NET":    ("Cloudflare", "USD"),
    "AXON":   ("Axon Enterprise", "USD"),
    "MELI":   ("MercadoLibre", "USD"),
    "TEAM":   ("Atlassian", "USD"),
    "MDB":    ("MongoDB", "USD"),
    "PSTG":   ("Pure Storage", "USD"),
    "HUBS":   ("HubSpot", "USD"),
    "OKTA":   ("Okta", "USD"),
    "SNOW":   ("Snowflake", "USD"),
    "PLTR":   ("Palantir", "USD"),
    "RBLX":   ("Roblox", "USD"),
    "U":      ("Unity Software", "USD"),
    "COIN":   ("Coinbase", "USD"),
    "SHOP":   ("Shopify", "USD"),
    "MSTR":  ("MicroStrategy", "USD"),
    "SIRI":   ("SiriusXM", "USD"),
    "LCID":   ("Lucid Motors", "USD"),
    "RIVN":   ("Rivian", "USD"),
    "NIO":    ("NIO", "USD"),
    # ── Semis & hardware ──
    "ON":     ("ON Semiconductor", "USD"),
    "SWKS":   ("Skyworks", "USD"),
    "QRVO":   ("Qorvo", "USD"),
    "SMCI":   ("Super Micro", "USD"),
    "MRAM":   ("Everspin", "USD"),
    "MPWR":   ("Monolithic Power", "USD"),
    "POWI":   ("Power Integrations", "USD"),
    # ── Biotech & pharma ──
    "MRNA":   ("Moderna", "USD"),
    "BNTX":   ("BioNTech", "USD"),
    "VTRS":   ("Viatris", "USD"),
    "DXCM":   ("Dexcom", "USD"),
    "ICLR":   ("Icon PLC", "USD"),
    "ALKS":   ("Alkermes", "USD"),
    # ── Fintech & payments ──
    "SQ":     ("Block", "USD"),
    "PATH":   ("UiPath", "USD"),
    "AFRM":   ("Affirm", "USD"),
    "UPST":   ("Upstart", "USD"),
    "HOOD":   ("Robinhood", "USD"),
    # ── Energy & industrials ──
    "ENPH":   ("Enphase Energy", "USD"),
    "SEDG":   ("SolarEdge", "USD"),
    "FSLR":   ("First Solar", "USD"),
    "GEHC":   ("GE HealthCare", "USD"),
    "PCAR":   ("PACCAR", "USD"),
    # ── European — Euronext Paris ──
    "MC.PA":  ("LVMH", "EUR"),
    "TTE.PA": ("TotalEnergies", "EUR"),
    "AIR.PA": ("Airbus", "EUR"),
    "SAN.PA": ("Sanofi", "EUR"),
    "BNP.PA": ("BNP Paribas", "EUR"),
    "ORA.PA": ("Orange", "EUR"),
    "EN.PA":  ("Engie", "EUR"),
    "RMS.PA": ("Hermes", "EUR"),
    "OR.PA":  ("L'Oreal", "EUR"),
    "CAP.PA": ("Capgemini", "EUR"),
    "SU.PA":  ("Schneider Electric", "EUR"),
    "ALO.PA": ("Alstom", "EUR"),
    "GLE.PA": ("Societe Generale", "EUR"),
    "ACA.PA": ("Credit Agricole", "EUR"),
    "DG.PA":  ("Vinci", "EUR"),
    "SGO.PA": ("Saint-Gobain", "EUR"),
    # ── European — Others ──
    "SAP.DE": ("SAP", "EUR"),
    "SIE.DE": ("Siemens", "EUR"),
    "ASML":   ("ASML", "EUR"),
    "NVO":    ("Novo Nordisk", "DKK"),
}

CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "stock_names.json"


def get_stock_info(symbol: str) -> tuple[str, str]:
    """
    Return (company_name, currency) for a ticker symbol.
    Uses local dictionary first, then falls back to yfinance, then to symbol itself.
    """
    # Check local dictionary first
    if symbol in STOCK_INFO:
        return STOCK_INFO[symbol]

    # Try cached file
    cached = _load_cache()
    if symbol in cached:
        return tuple(cached[symbol])

    # Try yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        name = info.get("shortName") or info.get("longName") or symbol
        currency = info.get("currency", "USD")
        result = (name, currency)
        # Cache it
        cached[symbol] = list(result)
        _save_cache(cached)
        return result
    except Exception as exc:
        logger.warning("yfinance lookup failed for %s: %s", symbol, exc)
        return (symbol, "USD")


def get_display_name(symbol: str) -> str:
    """Return 'Company Name (TICKER)' for display."""
    name, _ = get_stock_info(symbol)
    if name == symbol:
        return symbol
    return f"{name} ({symbol})"


def get_currency(symbol: str) -> str:
    """Return the currency for a ticker (EUR, USD, etc.)."""
    _, currency = get_stock_info(symbol)
    return currency


def format_price(value: float, currency: str = "EUR") -> str:
    """Format a price value with the correct currency symbol."""
    if currency == "EUR":
        return f"{value:,.2f} €"
    if currency == "GBP":
        return f"£{value:,.2f}"
    return f"${value:,.2f}"


def format_price_sign(value: float, currency: str = "EUR") -> str:
    """Format a price with sign (+/-) and correct currency symbol."""
    sign = "+" if value >= 0 else ""
    if currency == "EUR":
        return f"{sign}{value:,.2f} €"
    if currency == "GBP":
        return f"{sign}£{value:,.2f}"
    return f"{sign}${value:,.2f}"


def _load_cache() -> dict:
    """Load cached stock names from JSON file."""
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(data: dict) -> None:
    """Save cached stock names to JSON file."""
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to save stock name cache: %s", exc)


def get_all_tickers() -> list[str]:
    """Return all available tickers (from STOCK_INFO dictionary)."""
    return list(STOCK_INFO.keys())


def search_tickers(query: str, limit: int = 30) -> list[str]:
    """Search tickers by name or symbol prefix. Returns up to `limit` matches."""
    query = query.strip().upper()
    if not query:
        return list(STOCK_INFO.keys())[:limit]
    matches = []
    for sym, (name, _) in STOCK_INFO.items():
        if sym.startswith(query) or name.upper().startswith(query):
            matches.append(sym)
    if not matches:
        for sym, (name, _) in STOCK_INFO.items():
            if query in sym or query in name.upper():
                matches.append(sym)
    return matches[:limit]