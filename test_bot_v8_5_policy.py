"""Regression checks for the V8.5 macro policy bias layer."""

import numpy as np

from bot_v8_2 import (
    CAND_ATTACK_FRAC,
    CAND_IS_DEFENSE,
    CAND_IS_TRANSFER,
    CAND_TRANSFER_FRAC,
    _policy_bonus,
)


def _cand(*, attack=0.0, transfer=0.0, defense=0.0, is_transfer=0.0):
    feat = np.zeros(14, dtype=np.float32)
    feat[CAND_ATTACK_FRAC] = attack
    feat[CAND_TRANSFER_FRAC] = transfer
    feat[CAND_IS_DEFENSE] = defense
    feat[CAND_IS_TRANSFER] = is_transfer
    return feat


def test_opening_fortify_is_preferred_when_opening_is_threatened():
    bonus = _policy_bonus(
        "opening_fortify",
        _cand(defense=1.0),
        is_4p=True,
        is_opening=True,
        is_late=False,
        active_fronts=1,
        g_ratio=0.28,
        ship_lead=1.0,
        prod_lead=1.0,
        has_conversion=False,
        has_opportunity=False,
    )
    baseline = _policy_bonus(
        "v7_baseline",
        _cand(),
        is_4p=True,
        is_opening=True,
        is_late=False,
        active_fronts=1,
        g_ratio=0.28,
        ship_lead=1.0,
        prod_lead=1.0,
        has_conversion=False,
        has_opportunity=False,
    )
    assert bonus > baseline


def test_transfer_push_is_rewarded_when_it_carries_real_staging():
    bonus = _policy_bonus(
        "transfer_push",
        _cand(transfer=0.22, is_transfer=1.0),
        is_4p=False,
        is_opening=False,
        is_late=False,
        active_fronts=0,
        g_ratio=0.52,
        ship_lead=1.0,
        prod_lead=1.0,
        has_conversion=False,
        has_opportunity=False,
    )
    assert bonus > 0


def test_4p_conservation_beats_passive_baseline_under_multi_front_pressure():
    conservation = _policy_bonus(
        "4p_conservation",
        _cand(defense=1.0),
        is_4p=True,
        is_opening=False,
        is_late=False,
        active_fronts=2,
        g_ratio=0.24,
        ship_lead=0.95,
        prod_lead=0.92,
        has_conversion=False,
        has_opportunity=False,
    )
    baseline = _policy_bonus(
        "v7_baseline",
        _cand(),
        is_4p=True,
        is_opening=False,
        is_late=False,
        active_fronts=2,
        g_ratio=0.24,
        ship_lead=0.95,
        prod_lead=0.92,
        has_conversion=False,
        has_opportunity=False,
    )
    assert conservation > baseline
