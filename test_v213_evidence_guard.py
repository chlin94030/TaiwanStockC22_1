"""V21.3 adversarial tests. Synthetic outcomes are NOT alpha evidence."""
from dataclasses import replace
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from lab_config import Config
from lab_candidates import candidate_registry
from lab_approval import (reference_choice, is_reference, pre_tail_gate,
                          equivalent_training_pools)
from lab_search import choose_inner, record_choice
from lab_backtest import Case
from reassess_uploaded_v21 import review


def cfg():
    return Config(end='2026-10-21', min_train_months=12,
                  block_bootstrap_samples=25)


def test_negative_objective_must_allow_benchmark_as_reference():
    c = cfg()
    cs = candidate_registry(c)
    n = 24
    a = np.full((n, len(cs)), -0.08)
    g = {'net': a, 'floor': a, 'fills': np.ones_like(a),
         'benchmark': np.zeros(n), 'feature_coverage': np.ones_like(a)}
    choice, board = choose_inner({('ADV100', 5): g}, cs, 5,
                                  np.arange(n), c)
    assert len(board) == len(cs)
    assert is_reference(choice)
    assert choice['weights'] == {}


def test_positive_training_strategy_can_still_be_candidate():
    c = cfg()
    cs = candidate_registry(c)
    n = 24
    a = np.full((n, len(cs)), -0.03)
    a[:, 1] = .06
    g = {'net': a, 'floor': a, 'fills': np.ones_like(a),
         'benchmark': np.zeros(n), 'feature_coverage': np.ones_like(a)}
    choice, _ = choose_inner({('ADV100', 20): g}, cs, 20,
                             np.arange(n), c)
    assert not is_reference(choice)
    assert choice['candidate_index'] == 1


def test_training_selection_cannot_see_holdout():
    c = cfg()
    cs = candidate_registry(c)
    a = np.full((36, len(cs)), -.02)
    g = {'net': a.copy(), 'floor': a.copy(), 'fills': np.ones_like(a),
         'benchmark': np.zeros(36), 'feature_coverage': np.ones_like(a)}
    old, _ = choose_inner({('ADV100', 5): g}, cs, 5,
                           np.arange(24), c)
    g['net'][24:] = 1000000
    g['floor'][24:] = 1000000
    new, _ = choose_inner({('ADV100', 5): g}, cs, 5,
                           np.arange(24), c)
    assert old == new and is_reference(new)


def fake_case(benchmark=.03):
    names = ('liquidity20',)
    return Case(pd.Timestamp('2024-01-31'), np.array(['2330.TW']),
                np.array([[1.]]), np.array([1e9]), np.array([.04]),
                np.array([True]), np.array([-.02]), benchmark,
                .035, 0.0, pd.Timestamp('2024-02-01'),
                pd.Timestamp('2024-02-08'), names)


def test_reference_is_not_stock_selection_and_zero_excess_is_identity():
    rows = record_choice(reference_choice(), [fake_case()], [0], cfg())
    assert rows[0]['state'] == 'BENCHMARK_REFERENCE'
    assert rows[0]['selected'] == ['0050.TW']
    assert rows[0]['excess_0050'] == 0
    assert np.isnan(rows[0]['excess_pool'])


def test_reference_missing_benchmark_is_unknown_not_cash():
    rows = record_choice(reference_choice(), [fake_case(float('nan'))],
                         [0], cfg())
    assert np.isnan(rows[0]['net'])
    assert np.isnan(rows[0]['excess_0050'])


def prepared_outer(mean=.02, low=.005, pool_cov=1.0):
    return {'versus_0050': {'coverage': 1.0, 'mean': mean,
                           'block95_low': low, 'disjoint_complete': 9,
                           'disjoint95_low': .002},
            'versus_same_pool': {'coverage': pool_cov, 'mean': mean,
                                 'block95_low': low, 'disjoint_complete': 9,
                                 'disjoint95_low': .002},
            'extra_cost_stress_excess': .008}


def prepared_choice():
    return {'pool': 'ADV200', 'candidate_id': 'c123',
            'objective': .03, 'train_stress_excess': .02,
            'inner_block_means': [.015, .025, .02]}


def prepared_folds():
    return [{'status': 'EVALUATED', 'test': {'versus_0050': {'mean': .01}}}
            for _ in range(5)]


def test_good_pre_tail_evidence_allows_research_gate_not_orders():
    out = pre_tail_gate(prepared_choice(), prepared_outer(),
                        prepared_folds(), cfg())
    assert out['status'] == 'PRETAIL_RESEARCH_GATE_PASSED'
    assert out['automatic_orders_authorized'] is False
    assert out['final_tail_used'] is False


def test_negative_pre_tail_stops_research_promotion():
    out = pre_tail_gate(prepared_choice(), prepared_outer(low=-.1),
                        prepared_folds(), cfg())
    assert out['status'] == 'RESEARCH_CANDIDATE_NOT_PROMOTED'
    assert 'VERSUS_0050_BLOCK_95_NOT_POSITIVE' in out['failed_checks']


def test_partial_same_pool_blocks_not_manufactured():
    out = pre_tail_gate(prepared_choice(), prepared_outer(pool_cov=.8),
                        prepared_folds(), cfg())
    assert 'VERSUS_SAME_POOL_INSUFFICIENT_COVERAGE' in out['failed_checks']


def test_negative_outer_folds_block_the_candidate():
    folds = prepared_folds()
    for f in folds[:3]:
        f['test']['versus_0050']['mean'] = -.03
    out = pre_tail_gate(prepared_choice(), prepared_outer(), folds, cfg())
    assert 'OUTER_FOLD_DIRECTION_UNSTABLE' in out['failed_checks']


def test_no_alpha_reference_always_stays_unapproved():
    out = pre_tail_gate(reference_choice(), prepared_outer(),
                        prepared_folds(), cfg())
    assert out['status'] == 'NO_ACTIVE_ALPHA'
    assert out['fallback'] == 'REFERENCE_0050'


def test_equivalent_training_returns_are_not_unique_pool_proof():
    n = 12
    g = {'net': np.ones((n, 2)), 'floor': np.ones((n, 2)),
         'fills': np.ones((n, 2)), 'feature_coverage': np.ones((n, 2))}
    choice = {'pool': 'ADV100', 'candidate_index': 1, 'candidate_id': 'c123'}
    assert equivalent_training_pools({('ADV100', 20): g,
                                     ('ADV200', 20): copy.deepcopy(g)},
                                    choice, np.arange(n)) == ['ADV100', 'ADV200']
    h = copy.deepcopy(g)
    h['net'][-1, 1] = 1.01
    assert equivalent_training_pools({('ADV100', 20): g,
                                     ('ADV200', 20): h},
                                    choice, np.arange(n)) == ['ADV100']


def test_original_real_result_is_read_only_and_unpromoted():
    p = Path('/mnt/data/LAB_RESULTS.zip')
    if not p.exists():
        return
    result = review(p)
    assert result['existing_export_sha256_checked'] is True
    assert result['new_returns_computed'] is False
    assert set(result['models']) == {'5', '20', '60'}
    assert all(x['pre_tail_gate_reassessment_on_old_exposed_results']['status']
               == 'RESEARCH_CANDIDATE_NOT_PROMOTED'
               for x in result['models'].values())
