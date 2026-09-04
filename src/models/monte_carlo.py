"""NFL Monte Carlo Game Simulation Engine (10,000 Vectorized Iterations).

Simulates possession-by-possession outcomes using discrete NFL scoring distributions,
tempo/pace modeling, weather interaction, and opponent-adjusted EPA efficiency.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SimulationResult:
    home_team: str
    away_team: str
    num_iterations: int
    home_win_prob: float
    away_win_prob: float
    tie_prob: float
    mean_home_score: float
    mean_away_score: float
    median_home_score: float
    median_away_score: float
    mean_spread: float  # home - away
    median_spread: float
    std_spread: float
    mean_total: float  # home + away
    median_total: float
    std_total: float
    vegas_spread: float
    vegas_total: float
    spread_cover_prob_home: float
    spread_cover_prob_away: float
    over_prob: float
    under_prob: float
    over_under_pick: str
    over_under_edge: float  # diff from 50%
    teaser_prob_home_6pts: float  # covering spread + 6 pts
    teaser_prob_away_6pts: float
    key_numbers_probs: Dict[str, float]  # e.g. "3": 0.15, "7": 0.12
    top_exact_scores: List[Dict[str, Any]]
    spread_histogram: Dict[str, List[Any]]  # bins and frequencies for frontend charts
    totals_histogram: Dict[str, List[Any]]


class MonteCarloSimulator:
    """High-performance vectorized Monte Carlo simulator for NFL games."""

    def __init__(self, num_iterations: int = 10000, random_seed: Optional[int] = 42):
        self.num_iterations = num_iterations
        self.rng = np.random.default_rng(random_seed)

    def _calculate_drive_probabilities(
        self,
        net_off_epa: float,
        net_pass_epa: float,
        wind_speed: float = 7.0,
        is_dome: int = 0,
    ) -> np.ndarray:
        """Calculates discrete probabilities for drive outcomes:

        [0 pts, 2 pts (safety), 3 pts (FG), 6 pts (TD missed PAT), 7 pts (TD+PAT), 8 pts (TD+2pt)]
        """
        # Baseline modern NFL drive outcome frequencies
        # Modern NFL average: ~23.5 pts per team (~11.2 possessions per game)
        # Expected points per drive = 0.245*7 + 0.16*3 = 2.195 pts -> 11.2 * 2.195 = 24.5 pts
        base_td = 0.240
        base_fg = 0.155
        base_safety = 0.003

        # net_off_epa is on per-play scale (typically -0.15 to +0.20)
        td_adj = (net_off_epa * 0.28) + (net_pass_epa * 0.12)
        fg_adj = (net_off_epa * 0.10)

        # Weather effects: high wind reduces FG accuracy and deep passing TDs
        if is_dome == 0 and wind_speed >= 15.0:
            wind_penalty = (wind_speed - 15.0) * 0.004
            fg_adj -= wind_penalty * 1.5
            td_adj -= wind_penalty

        p_td = float(np.clip(base_td + td_adj, 0.12, 0.38))
        p_fg = float(np.clip(base_fg + fg_adj, 0.08, 0.24))
        p_safety = base_safety

        # TD point distribution: 92% get 7 pts, 5% get 8 pts (2-pt conv), 3% get 6 pts (missed PAT)
        p_td_7 = p_td * 0.92
        p_td_8 = p_td * 0.05
        p_td_6 = p_td * 0.03

        p_zero = max(0.0, 1.0 - (p_td_7 + p_td_8 + p_td_6 + p_fg + p_safety))

        # Outcome points: [0, 2, 3, 6, 7, 8]
        probs = np.array([p_zero, p_safety, p_fg, p_td_6, p_td_7, p_td_8], dtype=np.float64)
        probs /= probs.sum()  # normalize
        return probs

    def simulate_game(
        self,
        home_team: str,
        away_team: str,
        home_net_epa: float = 0.0,
        away_net_epa: float = 0.0,
        home_pass_epa: float = 0.0,
        away_pass_epa: float = 0.0,
        vegas_spread: float = 0.0,
        vegas_total: float = 44.5,
        star_spread_adj: float = 0.0,
        star_total_adj: float = 0.0,
        wind_speed: float = 7.0,
        is_dome: int = 0,
        temperature: float = 68.0,
    ) -> SimulationResult:
        """Executes 10,000 simulated games and derives full betting analytics."""
        N = self.num_iterations

        # 1. Pace & Possessions modeling
        # Base NFL possessions per game: ~11.2 per team
        base_poss = 11.2
        if is_dome == 0 and wind_speed >= 16.0:
            base_poss -= 0.4
        if temperature <= 32.0:
            base_poss -= 0.3

        # Possessions for Home and Away (correlated game pace)
        game_pace_jitter = self.rng.normal(0, 0.85, size=N)
        home_possessions = np.clip(np.round(base_poss + game_pace_jitter + self.rng.normal(0, 0.4, size=N)), 8, 16).astype(int)
        away_possessions = np.clip(np.round(base_poss + game_pace_jitter + self.rng.normal(0, 0.4, size=N)), 8, 16).astype(int)

        # 2. Drive Outcome Probabilities
        # Point outcomes corresponding to indices: [0, 2, 3, 6, 7, 8]
        pts_values = np.array([0, 2, 3, 6, 7, 8], dtype=np.int32)

        # Incorporate star player adjustments into net EPA
        h_eff = home_net_epa + (star_spread_adj * 0.05) + (star_total_adj * 0.02)
        a_eff = away_net_epa - (star_spread_adj * 0.05) + (star_total_adj * 0.02)

        home_probs = self._calculate_drive_probabilities(h_eff, home_pass_epa, wind_speed, is_dome)
        away_probs = self._calculate_drive_probabilities(a_eff, away_pass_epa, wind_speed, is_dome)

        # Max possessions matrix
        max_p = 16
        # Vectorized multinomial sampling for all N games and max possessions
        home_drive_pts = self.rng.choice(pts_values, size=(N, max_p), p=home_probs)
        away_drive_pts = self.rng.choice(pts_values, size=(N, max_p), p=away_probs)

        # Mask drives beyond each game's sampled possession count
        poss_indices = np.arange(max_p)
        home_mask = poss_indices < home_possessions[:, np.newaxis]
        away_mask = poss_indices < away_possessions[:, np.newaxis]

        home_scores = np.sum(home_drive_pts * home_mask, axis=1)
        away_scores = np.sum(away_drive_pts * away_mask, axis=1)

        # Overtime Resolution for ties (NFL regular season OT rules)
        tied_indices = np.where(home_scores == away_scores)[0]
        if len(tied_indices) > 0:
            # Overtime: 48% Home wins (FG or TD), 48% Away wins, 4% remains tie
            ot_outcomes = self.rng.choice([3, 6, -3, -6, 0], size=len(tied_indices), p=[0.24, 0.24, 0.24, 0.24, 0.04])
            for i, idx in enumerate(tied_indices):
                ot_pts = ot_outcomes[i]
                if ot_pts > 0:
                    home_scores[idx] += ot_pts
                elif ot_pts < 0:
                    away_scores[idx] += abs(ot_pts)

        # 3. Distributions & Statistical Summaries
        spreads = home_scores - away_scores  # Home margin
        totals = home_scores + away_scores

        home_wins = np.sum(spreads > 0)
        away_wins = np.sum(spreads < 0)
        ties = np.sum(spreads == 0)

        home_win_prob = round(float(home_wins / N * 100), 1)
        away_win_prob = round(float(away_wins / N * 100), 1)
        tie_prob = round(float(ties / N * 100), 1)

        mean_h = float(np.mean(home_scores))
        mean_a = float(np.mean(away_scores))
        mean_sp = float(np.mean(spreads))
        std_sp = float(np.std(spreads))
        mean_tot = float(np.mean(totals))
        std_tot = float(np.std(totals))

        # 4. Vegas Line Analysis & Cover Probabilities
        # vegas_spread from perspective of Home (e.g. -3.5 means Home favored by 3.5)
        # Home covers if spreads > (-vegas_spread)
        h_covers = np.sum(spreads > (-vegas_spread))
        a_covers = np.sum(spreads < (-vegas_spread))
        spread_cover_h = round(float(h_covers / N * 100), 1)
        spread_cover_a = round(float(a_covers / N * 100), 1)

        # Over / Under
        overs = np.sum(totals > vegas_total)
        unders = np.sum(totals < vegas_total)
        over_prob = round(float(overs / N * 100), 1)
        under_prob = round(float(unders / N * 100), 1)

        ou_edge = round(abs(over_prob - 50.0), 1)
        if over_prob >= 53.5:
            ou_pick = f"OVER {vegas_total} pts ({over_prob}% prob)"
        elif under_prob >= 53.5:
            ou_pick = f"UNDER {vegas_total} pts ({under_prob}% prob)"
        else:
            ou_pick = f"NEUTRAL / PASS ({vegas_total} pts)"

        # Teaser probabilities (+6.0 pts)
        teaser_h = round(float(np.sum(spreads > (-vegas_spread - 6.0)) / N * 100), 1)
        teaser_a = round(float(np.sum(spreads < (-vegas_spread + 6.0)) / N * 100), 1)

        # 5. Key Numbers Analysis (Abs Margin)
        abs_margins = np.abs(spreads)
        key_nums = [1, 2, 3, 4, 6, 7, 10, 14]
        key_numbers_probs = {}
        for kn in key_nums:
            key_numbers_probs[str(kn)] = round(float(np.sum(abs_margins == kn) / N * 100), 2)

        # 6. Top Exact Scores
        unique_scores, counts = np.unique(
            np.column_stack((home_scores, away_scores)), axis=0, return_counts=True
        )
        sort_order = np.argsort(-counts)
        top_exact = []
        for idx in sort_order[:6]:
            h_s, a_s = unique_scores[idx]
            cnt = counts[idx]
            top_exact.append({
                "score": f"{home_team} {h_s} - {a_s} {away_team}",
                "home_score": int(h_s),
                "away_score": int(a_s),
                "prob": round(float(cnt / N * 100), 2),
            })

        # 7. Histogram Bins for Frontend Charts
        # Spread bins from -28 to +28 in steps of 3
        sp_bins = np.arange(-28, 32, 4)
        sp_hist, _ = np.histogram(spreads, bins=sp_bins)
        spread_histogram = {
            "labels": [f"{sp_bins[i]} to {sp_bins[i+1]}" for i in range(len(sp_bins)-1)],
            "counts": [int(c) for c in sp_hist],
        }

        # Totals bins from 24 to 68 in steps of 4
        tot_bins = np.arange(24, 72, 4)
        tot_hist, _ = np.histogram(totals, bins=tot_bins)
        totals_histogram = {
            "labels": [f"{tot_bins[i]} to {tot_bins[i+1]}" for i in range(len(tot_bins)-1)],
            "counts": [int(c) for c in tot_hist],
        }

        return SimulationResult(
            home_team=home_team,
            away_team=away_team,
            num_iterations=N,
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            tie_prob=tie_prob,
            mean_home_score=round(mean_h, 1),
            mean_away_score=round(mean_a, 1),
            median_home_score=float(np.median(home_scores)),
            median_away_score=float(np.median(away_scores)),
            mean_spread=round(mean_sp, 1),
            median_spread=float(np.median(spreads)),
            std_spread=round(std_sp, 1),
            mean_total=round(mean_tot, 1),
            median_total=float(np.median(totals)),
            std_total=round(std_tot, 1),
            vegas_spread=vegas_spread,
            vegas_total=vegas_total,
            spread_cover_prob_home=spread_cover_h,
            spread_cover_prob_away=spread_cover_a,
            over_prob=over_prob,
            under_prob=under_prob,
            over_under_pick=ou_pick,
            over_under_edge=ou_edge,
            teaser_prob_home_6pts=teaser_h,
            teaser_prob_away_6pts=teaser_a,
            key_numbers_probs=key_numbers_probs,
            top_exact_scores=top_exact,
            spread_histogram=spread_histogram,
            totals_histogram=totals_histogram,
        )
