"""
TradeFlow — Strategy Optimizer
Grid search hyperparameter tuning for trading strategies.
"""

import itertools
import logging
from typing import Any, Callable, Generator

from app.simulator.engine import SimResult, SimulationEngine
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def grid_search(
    strategy_cls: type[BaseStrategy],
    param_grid: dict[str, list[Any]],
    symbol: str,
    interval: str = "1h",
    period: str = "1y",
    initial_capital: float = 10_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[SimResult]:
    """
    Run a grid search optimization over a parameter grid.
    
    Args:
        strategy_cls: The strategy class to optimize.
        param_grid: Dictionary mapping parameter names to lists of values to test.
        symbol: Asset ticker.
        interval: Bar interval.
        period: Historical data period.
        initial_capital: Starting capital.
        commission_rate: Broker commission.
        slippage_rate: Broker slippage.
        progress_callback: Callable(current_step, total_steps, current_params) for UI updates.
        
    Returns:
        List of SimResult objects for each tested combination, sorted by Total Return (descending).
    """
    # Generate all combinations of parameters
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    combinations = list(itertools.product(*value_lists))
    
    total_runs = len(combinations)
    logger.info("Starting grid search for %s: %d combinations", strategy_cls.__name__, total_runs)
    
    # We use a single engine for all runs
    engine = SimulationEngine(
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )
    
    results: list[SimResult] = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        if progress_callback:
            progress_callback(i + 1, total_runs, params)
            
        try:
            # Instantiate strategy with current parameters
            strategy = strategy_cls(**params)
            
            # Run without saving to DB to preserve speed and avoid DB pollution
            result = engine.run(
                strategy=strategy,
                symbol=symbol,
                interval=interval,
                period=period,
                initial_capital=initial_capital,
                save_to_db=False,
            )
            
            if result:
                # Attach the parameters to the result for easy access
                setattr(result, "params", params)
                results.append(result)
                
        except Exception as e:
            logger.error("Simulation failed for params %s: %s", params, e)
            
    # Sort results by Total Return (descending)
    results.sort(key=lambda r: r.total_return_pct, reverse=True)
    return results
