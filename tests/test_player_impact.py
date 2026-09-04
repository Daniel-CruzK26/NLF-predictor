"""Unit tests for player impact & injury adjustments."""

from src.features.player_impact import (
    calculate_matchup_star_impact,
    get_team_star_ratings,
    load_star_players,
)


def test_load_star_players():
    catalog = load_star_players()
    assert len(catalog) == 32
    assert "KC" in catalog
    assert "SF" in catalog
    assert any(p["name"] == "Travis Kelce" for p in catalog["KC"])


def test_team_star_ratings_and_injuries():
    catalog = {
        "TEST": [
            {"name": "Star WR", "pos": "WR", "unit": "offense", "value_pts": 2.0},
            {"name": "Star EDGE", "pos": "EDGE", "unit": "defense", "value_pts": 1.5},
        ]
    }
    # All active
    ratings_active = get_team_star_ratings("TEST", stars_catalog=catalog, injuries={})
    assert ratings_active["off_stars_active"] == 2.0
    assert ratings_active["def_stars_active"] == 1.5
    assert ratings_active["off_stars_lost"] == 0.0

    # Star WR OUT, EDGE QUESTIONABLE
    injuries = {"Star WR": "OUT", "Star EDGE": "QUESTIONABLE"}
    ratings_injured = get_team_star_ratings("TEST", stars_catalog=catalog, injuries=injuries)
    assert ratings_injured["off_stars_active"] == 0.0
    assert ratings_injured["off_stars_lost"] == 2.0
    assert ratings_injured["def_stars_active"] == 0.75
    assert ratings_injured["def_stars_lost"] == 0.75


def test_matchup_star_impact():
    catalog = {
        "HOME": [{"name": "H_WR", "pos": "WR", "unit": "offense", "value_pts": 2.0}],
        "AWAY": [{"name": "A_CB", "pos": "CB", "unit": "defense", "value_pts": 1.0}],
    }
    impact = calculate_matchup_star_impact("HOME", "AWAY", stars_catalog=catalog, injuries={})
    assert "star_spread_adjustment" in impact
    assert "star_total_adjustment" in impact
    # Home has offensive star advantage (+2 vs +0), so spread adjustment should favor Home (> 0)
    assert impact["star_spread_adjustment"] > 0
