"""CLI Script to run 10,000-iteration Monte Carlo simulation for upcoming NFL slate."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_schedules
from src.features.player_impact import calculate_matchup_star_impact
from src.models.inference import LiveInferenceEngine
from src.models.monte_carlo import MonteCarloSimulator


def main():
    parser = argparse.ArgumentParser(description="Run 10,000-iteration Monte Carlo NFL Game Simulation.")
    parser.add_argument("--season", type=int, default=2026, help="Season (default: 2026)")
    parser.add_argument("--week", type=int, default=1, help="Week (default: 1)")
    parser.add_argument("--matchup", type=str, default=None, help="Optional specific matchup e.g. 'DEN_KC'")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of simulations (default: 10000)")
    args = parser.parse_args()

    print(f"🎲 [1/3] Initializing Monte Carlo Engine ({args.iterations:,} iterations)...")
    simulator = MonteCarloSimulator(num_iterations=args.iterations)

    engine = LiveInferenceEngine()
    engine.fit_production_models()

    schedules = load_schedules([args.season])
    week_games = schedules[schedules["week"] == args.week].copy()

    if args.matchup:
        parts = args.matchup.split("_")
        if len(parts) == 2:
            week_games = week_games[(week_games["away_team"] == parts[0]) & (week_games["home_team"] == parts[1])]

    print(f"🏈 [2/3] Simulating {len(week_games)} game(s) for Season {args.season} Week {args.week}...\n")

    print(f"{'Matchup':<12} | {'Sim Score':<14} | {'Spread (H-A)':<12} | {'Total':<10} | {'Win Prob (H/A)':<16} | {'Over/Under Pick':<26}")
    print("-" * 105)

    results = []
    for _, row in week_games.iterrows():
        h_t = row["home_team"]
        a_t = row["away_team"]
        v_sp = float(row.get("spread_line", 0.0))
        v_tot = float(row.get("total_line", 44.5)) if pd.notna(row.get("total_line")) else 44.5

        # Extract per-play EPA priors
        h_priors = engine.team_priors.get(h_t, {})
        a_priors = engine.team_priors.get(a_t, {})

        h_pass_net = h_priors.get("adj_off_pass_epa", 0.0) - a_priors.get("adj_def_pass_epa_allowed", 0.0)
        h_rush_net = h_priors.get("adj_off_rush_epa", 0.0) - a_priors.get("adj_def_rush_epa_allowed", 0.0)
        h_net_epa = (0.60 * h_pass_net) + (0.40 * h_rush_net)

        a_pass_net = a_priors.get("adj_off_pass_epa", 0.0) - h_priors.get("adj_def_pass_epa_allowed", 0.0)
        a_rush_net = a_priors.get("adj_off_rush_epa", 0.0) - h_priors.get("adj_def_rush_epa_allowed", 0.0)
        a_net_epa = (0.60 * a_pass_net) + (0.40 * a_rush_net)

        # Star player impact
        star_impact = calculate_matchup_star_impact(h_t, a_t)

        sim_res = simulator.simulate_game(
            home_team=h_t,
            away_team=a_t,
            home_net_epa=h_net_epa,
            away_net_epa=a_net_epa,
            home_pass_epa=h_pass_net,
            away_pass_epa=a_pass_net,
            vegas_spread=v_sp,
            vegas_total=v_tot,
            star_spread_adj=star_impact["star_spread_adjustment"],
            star_total_adj=star_impact["star_total_adjustment"],
            wind_speed=float(row.get("wind", 7.0) or 7.0),
            is_dome=1 if row.get("roof") in ["dome", "closed"] else 0,
        )

        matchup_str = f"{a_t} @ {h_t}"
        score_str = f"{sim_res.mean_away_score:.1f} - {sim_res.mean_home_score:.1f}"
        spread_str = f"{sim_res.mean_spread:+.1f} pts"
        tot_str = f"{sim_res.mean_total:.1f} pts"
        prob_str = f"{sim_res.home_win_prob:.1f}% / {sim_res.away_win_prob:.1f}%"

        print(f"{matchup_str:<12} | {score_str:<14} | {spread_str:<12} | {tot_str:<10} | {prob_str:<16} | {sim_res.over_under_pick:<26}")
        results.append(sim_res)

    print("\n✅ Monte Carlo simulations finished successfully!")


if __name__ == "__main__":
    main()
