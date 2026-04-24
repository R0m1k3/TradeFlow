"""
TradeFlow — Human-Readable French Explanations
Converts 0-1 composite scores into plain French text for beginners.
"""


def explain_score(score: float) -> str:
    """Return a plain French explanation of what the composite score means."""
    if score >= 0.85:
        return "Les indicateurs sont tres favorables, c'est le moment d'acheter"
    if score >= 0.7:
        return "Le marche montre des signaux positifs, on peut acheter"
    if score >= 0.55:
        return "Legerement positif, mais pas assez pour acheter en confiance"
    if score >= 0.45:
        return "Pas de signal clair, on attend de voir"
    if score >= 0.3:
        return "Le marche hesite, les signaux sont legerement negatifs"
    if score >= 0.15:
        return "Les indicateurs sont defavorables, mieux vaut vendre"
    return "Le marche est tres pessimiste, signal fort de vente"


def signal_label(score: float) -> str:
    """Return a short French action label based on score."""
    if score >= 0.7:
        return "Acheter"
    if score <= 0.3:
        return "Vendre"
    return "Attendre"


def signal_badge_class(score: float) -> str:
    """Return CSS class name for the badge color."""
    if score >= 0.7:
        return "tf-badge-buy"
    if score <= 0.3:
        return "tf-badge-sell"
    return "tf-badge-hold"


def card_class(score: float) -> str:
    """Return CSS class for the card top-border color."""
    if score >= 0.7:
        return "buy"
    if score <= 0.3:
        return "sell"
    return "hold"


def score_color(score: float) -> str:
    """Return hex color for a 0-1 score."""
    if score >= 0.7:
        return "#00C896"
    if score <= 0.3:
        return "#FF4B6E"
    return "#FFB347"


def pnl_color(value: float) -> str:
    """Return hex color for a P&L value (green if positive, red if negative)."""
    return "#00C896" if value >= 0 else "#FF4B6E"


def pnl_class(value: float) -> str:
    """Return CSS class for P&L (positive/negative)."""
    return "positive" if value >= 0 else "negative"


def format_pnl(value: float) -> str:
    """Format a P&L value with sign and currency."""
    sign = "+" if value >= 0 else ""
    return f"{sign}${value:,.2f}"


def explain_sub_score(label: str, value: float) -> str:
    """Return a plain French explanation for a sub-score."""
    explanations = {
        "Technique": {
            (0.7, 1.0): "Les graphiques montrent une tendance haussiere",
            (0.45, 0.7): "Pas de tendance claire sur les graphiques",
            (0.0, 0.45): "Les graphiques montrent une tendance baissiere",
        },
        "Sentiment": {
            (0.7, 1.0): "Les actualites et le marche sont optimistes",
            (0.45, 0.7): "L'humeur du marche est neutre",
            (0.0, 0.45): "Les actualites et le marche sont pessimistes",
        },
        "Momentum": {
            (0.7, 1.0): "Le volume d'echange confirme la hausse",
            (0.45, 0.7): "Le volume d'echange est normal",
            (0.0, 0.45): "Le volume d'echange confirme la baisse",
        },
    }
    ranges = explanations.get(label, {})
    for (lo, hi), text in ranges.items():
        if lo <= value <= hi:
            return text
    return ""