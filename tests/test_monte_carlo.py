"""Unit tests for Monte Carlo NFL game simulator."""

import time
from src.models.monte_carlo import MonteCarloSimulator, SimulationResult


def test_monte_carlo_execution_and_speed():
    sim = MonteCarloSimulator(num_iterations=10000, random_seed=42)

    t0 = time.time()
    result = sim.simulate_game(
        home_team="KC",
        away_team="DEN",
        home_net_epa=0.08,
        away_net_epa=-0.04,
        vegas_spread=3.0,
        vegas_total=44.5,
    )
    elapsed = time.time() - t0

    # Must be faster than 250ms for 10,000 iterations
    assert elapsed < 0.25, f"Simulation took too long: {elapsed:.3f}s"
    assert isinstance(result, SimulationResult)
    assert result.num_iterations == 10000


def test_monte_carlo_statistical_plausibility():
    sim = MonteCarloSimulator(num_iterations=10000, random_seed=42)

    result = sim.simulate_game(
        home_team="DET",
        away_team="NO",
        home_net_epa=0.15,
        away_net_epa=-0.10,
        vegas_spread=7.0,
        vegas_total=46.5,
    )

    # Typical NFL scores are between 15 and 35 points
    assert 18.0 <= result.mean_home_score <= 34.0
    assert 12.0 <= result.mean_away_score <= 28.0
    assert 35.0 <= result.mean_total <= 58.0

    # Strong home favorite should have higher win probability
    assert result.home_win_prob > result.away_win_prob

    # Key numbers 3 and 7 should have significant probabilities
    assert result.key_numbers_probs["3"] > 5.0
    assert result.key_numbers_probs["7"] > 4.0

    # Check top exact scores length
    assert len(result.top_exact_scores) >= 5
    assert len(result.spread_histogram["counts"]) > 0
    assert len(result.totals_histogram["counts"]) > 0


def test_monte_carlo_star_and_weather_impact():
    sim = MonteCarloSimulator(num_iterations=10000, random_seed=42)

    # Baseline game
    base_res = sim.simulate_game("BUF", "MIA", wind_speed=5.0, is_dome=0)

    # High wind game (wind reduces scoring and totals)
    wind_res = sim.simulate_game("BUF", "MIA", wind_speed=25.0, is_dome=0)

    assert wind_res.mean_total < base_res.mean_total
