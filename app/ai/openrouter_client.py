"""OpenRouter API client — kept as a thin compatibility shim.

The heavy lifting now lives in `app.ai.provider`. This module re-exports the
historical names so existing imports keep working, while routing through the
unified client (which adds caching, retries, and provider-agnostic parsing).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import httpx

from app.ai import provider as _provider

logger = logging.getLogger(__name__)

ENDPOINT = _provider.OPENROUTER_ENDPOINT
MODELS_ENDPOINT = _provider.OPENROUTER_MODELS_ENDPOINT

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


def fetch_models(api_key: str = "", timeout: int = 10) -> list[str]:
    """Return sorted list of model IDs available on OpenRouter."""
    return _provider.list_openrouter_models(api_key=api_key, timeout=timeout)


async def fetch_ai_score(
    ticker: str,
    model: str,
    api_key: str,
    timeout: int = 30,
) -> tuple[float, str, list]:
    """Backwards-compatible async helper. Returns (score, rationale, sources)."""
    from app.ai.provider import AIConfig
    cfg = AIConfig(
        provider="openrouter",
        model=model,
        api_key=api_key,
        timeout_seconds=timeout,
        cache_ttl_seconds=0,  # don't cache async helper calls — caller decides
    )
    prompt = PROMPT_TEMPLATE.format(ticker=ticker, date=datetime.now().strftime("%Y-%m-%d %H:%M"))
    try:
        result = await _call_async(prompt, cfg)
    except Exception:
        # Fallback: keep historical behaviour — raise on error
        raise
    parsed = result.get("json") or {}
    score = float(parsed.get("score", 0.5))
    score = max(0.0, min(1.0, score))
    rationale = parsed.get("rationale", "") or ""
    sources = parsed.get("sources") or []
    if not isinstance(sources, list):
        sources = []
    return score, rationale, sources


async def _call_async(prompt: str, cfg) -> dict:
    """Run the unified provider in a thread so it doesn't block the event loop."""
    import asyncio
    return await asyncio.to_thread(_provider.call_ai, prompt, mode="hybrid", cfg=cfg)


async def test_connection(api_key: str, model: str, timeout: int = 15) -> bool:
    """Quick connectivity check (legacy signature). Returns True if OK."""
    res = _provider.test_connection("openrouter", api_key, model=model, timeout=timeout)
    return bool(res.get("ok"))


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
