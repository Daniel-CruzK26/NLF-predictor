"""Script 06: Live prediction and betting edge detection for any NFL week."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR
from src.data.loader import load_schedules, standardize_team_abbr
from src.models.inference import LiveInferenceEngine, GamePrediction


from src.config_rosters import get_active_starters


def main():
    parser = argparse.ArgumentParser(description="Predict upcoming NFL week slate and detect betting edges.")
    parser.add_argument("--season", type=int, default=2026, help="Season to predict (default: 2026)")
    parser.add_argument("--week", type=int, default=1, help="Week to predict (default: 1)")
    parser.add_argument("--min-edge", type=float, default=1.5, help="Minimum edge to highlight bets (default: 1.5 pts)")
    args = parser.parse_args()

    print(f"🏈 [1/3] Loading schedule slate for Season {args.season} Week {args.week}...")
    schedules = load_schedules([args.season])
    week_games = schedules[schedules["week"] == args.week].copy()

    if len(week_games) == 0:
        print(f"❌ Error: No games found for Season {args.season} Week {args.week}.")
        sys.exit(1)

    print(f"   ✓ Found {len(week_games)} scheduled games.")

    # Assign Active Starting QBs (dynamically from rosters config)
    active_starters = get_active_starters()
    home_qbs = []
    away_qbs = []
    for _, row in week_games.iterrows():
        h_t = row["home_team"]
        a_t = row["away_team"]
        home_qbs.append(active_starters.get(h_t, f"{h_t}_QB"))
        away_qbs.append(active_starters.get(a_t, f"{a_t}_QB"))

    week_games["home_qb_name"] = home_qbs
    week_games["away_qb_name"] = away_qbs

    print("🤖 [2/3] Initializing Live Inference Engine and fitting production models...")
    engine = LiveInferenceEngine()
    engine.fit_production_models()

    print("🎯 [3/3] Generating Spread Projections, Win Probabilities, and Value Bets (+EV)...")
    predictions = engine.predict_slate(week_games, edge_threshold_low=args.min_edge)

    pred_dicts = [p.to_dict() for p in predictions]
    results_df = pd.DataFrame(pred_dicts)

    # Save to CSV and Parquet
    csv_out = PROCESSED_DATA_DIR / f"predictions_{args.season}_week_{args.week}.csv"
    parquet_out = PROCESSED_DATA_DIR / f"predictions_{args.season}_week_{args.week}.parquet"
    results_df.to_csv(csv_out, index=False)
    results_df.to_parquet(parquet_out, index=False)

    # Display Detailed Terminal Output
    print("\n" + "=" * 105)
    print(f"🏈 NFL PREDICTIONS & BETTING EDGES — SEASON {args.season} WEEK {args.week}")
    print("=" * 105)
    
    header = f"{'Matchup':<18} | {'Date':<10} | {'Vegas':<7} | {'Model':<7} | {'Win Prob (H/A)':<15} | {'Edge':<7} | {'Recommendation':<20} | {'Confidence':<10}"
    print(header)
    print("-" * 105)

    for p in predictions:
        matchup_str = f"{p.away_team} @ {p.home_team}"
        v_str = f"{p.vegas_spread:+.1f}"
        m_str = f"{p.model_spread:+.1f}"
        prob_str = f"{p.home_win_prob*100:.0f}% / {p.away_win_prob*100:.0f}%"
        edge_str = f"{p.edge:+.1f} pts"
        
        line_out = f"{matchup_str:<18} | {p.gameday[:10]:<10} | {v_str:<7} | {m_str:<7} | {prob_str:<15} | {edge_str:<7} | {p.recommendation:<20} | {p.confidence:<10}"
        print(line_out)

    # Highlight High-Value Picks (+EV)
    value_bets = [p for p in predictions if p.confidence != "NO VALUE"]
    print("\n" + "=" * 105)
    print(f"🔥 TOP VALUE PICKS (+EV BETS) — {len(value_bets)} OPPORTUNITIES DETECTED (Edge >= {args.min_edge} pts)")
    print("=" * 105)

    if value_bets:
        for rank, p in enumerate(sorted(value_bets, key=lambda x: abs(x.edge), reverse=True), start=1):
            print(f"\n  #{rank}. {p.recommendation} — Confidence: {p.confidence}")
            print(f"      • Matchup:        {p.away_team} ({p.away_qb}) @ {p.home_team} ({p.home_qb})")
            print(f"      • Spread Line:    Vegas {p.vegas_spread:+.1f} pts  vs  Model Proj {p.model_spread:+.1f} pts")
            print(f"      • Edge Discrepancy: {p.edge:+.2f} points of value")
            print(f"      • Win Probability:  Home ({p.home_team}) {p.home_win_prob*100:.1f}% | Away ({p.away_team}) {p.away_win_prob*100:.1f}%")
            print(f"      • Kelly Stake:      {p.kelly_stake_pct:.1f}% of Bankroll (1/4 Kelly)")
            print(f"      • Key Factors:      {' | '.join(p.key_drivers)}")
    else:
        print("  No significant discrepancies detected against Las Vegas lines for this slate.")

    print(f"\n✅ Predictions exported to:\n   - {csv_out}\n   - {parquet_out}")


if __name__ == "__main__":
    main()
