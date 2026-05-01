from neural_network.src.encoder import encode_game_state


def sample_game():
    return {
        "my_id": 0,
        "player_ids": [0, 1],
        "turn": 0,
        "planets": [
            {"id": 0, "owner": 0, "x": 10.0, "y": 20.0, "radius": 1.0, "production": 2.0, "ships": 30.0},
            {"id": 1, "owner": -1, "x": 30.0, "y": 40.0, "radius": 1.0, "production": 3.0, "ships": 20.0},
        ],
        "fleets": [
            {"owner": 1, "x": 12.0, "y": 22.0, "ships": 5.0, "eta": 3, "source_id": 1, "target_id": 0},
        ],
        "is_four_player": False,
    }


def test_encoder_produces_vector_and_masks():
    cfg = {"max_planets": 4, "max_fleets": 4, "max_players": 4, "board_scale": 100.0, "ship_scale": 200.0, "production_scale": 10.0, "radius_scale": 10.0, "horizon_scale": 100.0, "planet_id_scale": 64.0}
    enc = encode_game_state(sample_game(), cfg)
    assert enc.features.ndim == 1
    assert enc.planet_mask.shape == (4,)
    assert enc.fleet_mask.shape == (4,)
    assert enc.player_mask.shape == (4,)
    assert enc.planet_mask[0] == 1.0

