"""OpenRouter API client — fetches an AI sentiment score for a ticker."""
import json
import logging
import httpx

logger = logging.getLogger(__name__)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"


def fetch_models(api_key: str = "", timeout: int = 10) -> list[str]:
    """Return sorted list of model IDs available on OpenRouter.

    Works without an API key (public endpoint) but the key improves rate limits.
    Returns an empty list on error.
    """
    headers = {"HTTP-Referer": "https://tradeflow.local", "X-Title": "TradeFlow"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(MODELS_ENDPOINT, headers=headers)
            r.raise_for_status()
        models = r.json().get("data", [])
        return sorted(m["id"] for m in models if "id" in m)
    except Exception as exc:
        logger.warning("Could not fetch OpenRouter model list: %s", exc)
        return []

PROMPT_TEMPLATE = """\
Ticker boursier : {ticker}
Date et heure : {date}

Analyse le comportement probable de cette action sur les prochaines 24-48 heures en te basant \
sur les dernières actualités, la tendance du secteur, et le sentiment du marché.

Réponds UNIQUEMENT avec un JSON valide, sans texte avant ou après :
{{
  "score": <float entre 0.0 et 1.0>,
  "rationale": "<analyse concise en français, 2-3 phrases>",
  "sources": [
    {{"title": "<titre de la source>", "url": "<url complète>"}},
    ...
  ]
}}

0.0 = fort signal de vente, 0.5 = neutre, 1.0 = fort signal d'achat.
Inclure 2 à 4 sources réelles et vérifiables utilisées pour l'analyse.
"""


async def fetch_ai_score(
    ticker: str,
    model: str,
    api_key: str,
    timeout: int = 30,
) -> tuple[float, str, list]:
    """Return (score 0-1, rationale, sources). Raises on error."""
    from datetime import datetime
    prompt = PROMPT_TEMPLATE.format(ticker=ticker, date=datetime.now().strftime("%Y-%m-%d %H:%M"))

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://tradeflow.local",
                "X-Title": "TradeFlow",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    data = json.loads(content)
    score = float(data["score"])
    score = max(0.0, min(1.0, score))
    rationale = data.get("rationale", "")
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    return score, rationale, sources


async def test_connection(api_key: str, model: str, timeout: int = 15) -> bool:
    """Quick connectivity check. Returns True if the API responds correctly."""
    try:
        score, _ = await fetch_ai_score("AAPL", model, api_key, timeout=timeout)
        return 0.0 <= score <= 1.0
    except Exception as exc:
        logger.warning("OpenRouter connection test failed: %s", exc)
        return False
