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

Consulte également les publications récentes sur X (Twitter/𝕏) concernant ${ticker} : \
posts d'analystes, dirigeants, investisseurs institutionnels et sentiment retail en temps réel. \
Ces données sociales complètent les actualités financières traditionnelles.

Réponds UNIQUEMENT avec un JSON valide, sans texte avant ou après :
{{
  "score": <float entre 0.0 et 1.0>,
  "rationale": "<analyse concise en français, 2-3 phrases incluant le sentiment X si pertinent>",
  "sources": [
    {{"title": "<titre de la source>", "url": "<url complète>"}},
    ...
  ]
}}

0.0 = fort signal de vente, 0.5 = neutre, 1.0 = fort signal d'achat.
Inclure 2 à 5 sources réelles et vérifiables (articles financiers, posts X, communiqués), \
avec URLs directes quand disponibles.
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
        score, _r, _s = await fetch_ai_score("AAPL", model, api_key, timeout=timeout)
        return 0.0 <= score <= 1.0
    except Exception as exc:
        logger.warning("OpenRouter connection test failed: %s", exc)
        return False


AUTONOMOUS_PROMPT_TEMPLATE = """\
Tu es ARIA (Autonomous Research & Investment Agent), un agent de trading algorithmique expert \
avec plus de 20 ans d'expérience sur les marchés financiers mondiaux \
(NYSE, NASDAQ, Euronext, LSE, TSE).

TICKER ANALYSÉ : {ticker}
DATE ET HEURE  : {date}

SOURCES À CONSULTER OBLIGATOIREMENT :
1. Actualités financières récentes (Bloomberg, Reuters, FT, WSJ — dernières 48h)
2. Publications récentes sur X (Twitter/𝕏) : analystes, dirigeants, institutionnels, retail
3. Indicateurs techniques : tendance, momentum, niveaux support/résistance
4. Données fondamentales : valorisation, résultats récents, guidances, dividendes
5. Contexte sectoriel et macro-économique (taux, USD, matières premières)
6. Flux institutionnels si disponibles (options flow, signaux dark pool)

PROCESSUS DE DÉCISION (suis ces étapes dans l'ordre) :
1. Analyser le contexte macro et sectoriel
2. Évaluer le momentum et la structure technique
3. Intégrer le sentiment (news + X/𝕏)
4. Identifier les catalyseurs proches (earnings, produits, événements réglementaires)
5. Calculer le ratio risque/rendement
6. Formuler une recommandation ferme

RÈGLES DE GESTION DU RISQUE (OBLIGATOIRES — ne pas déroger) :
- confidence < 0.50 → action = "HOLD", position_size_pct = 0
- confidence 0.50–0.65 → position_size_pct ≤ 2 %, ratio R/R ≥ 1:2
- confidence 0.65–0.80 → position_size_pct ≤ 5 %, ratio R/R ≥ 1:2
- confidence > 0.80 → position_size_pct ≤ 10 %, ratio R/R ≥ 1:2.5
- take_profit_pct DOIT être ≥ 2 × stop_loss_pct

Réponds UNIQUEMENT avec un JSON valide, sans texte avant ni après :
{{
  "action": "<BUY | SELL | HOLD>",
  "confidence": <float 0.0–1.0>,
  "position_size_pct": <float 0.0–10.0>,
  "stop_loss_pct": <float 0.5–15.0>,
  "take_profit_pct": <float 1.0–30.0>,
  "time_horizon": "<ex : 24h | 48h | 1 semaine>",
  "rationale": "<analyse détaillée en français, 4–6 phrases, inclure données X si pertinentes>",
  "key_risks": "<2–3 risques principaux à surveiller, concis>",
  "sources": [
    {{"title": "<titre exact>", "url": "<url complète>"}},
    ...
  ]
}}

Inclure 3 à 6 sources réelles, vérifiables et récentes (< 48h si possible). Posts X acceptés.
"""
