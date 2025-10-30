#!/usr/bin/env python3
"""Quick import/runtime check for updated Python modules.

This script imports key modules and exercises two-layer NN paths to
surface any syntax/runtime errors introduced by recent edits.
"""
import sys
import os
import json

# ensure repo root is on sys.path so package imports (webviewer, scripts) work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
scripts_dir = os.path.join(repo_root, 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

def fail(msg, exc=None):
    print("ERROR:", msg)
    if exc:
        import traceback
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    sys.exit(2)

try:
    from webviewer import graphs_nn as g_nn
    print('imported webviewer.graphs_nn')
except Exception as e:
    fail('could not import webviewer.graphs_nn', e)

try:
    import scripts.classify_strategies_global as csg
    print('imported scripts.classify_strategies_global')
except Exception as e:
    fail('could not import scripts.classify_strategies_global', e)

try:
    import scripts.classify_and_plot_categories as capc
    print('imported scripts.classify_and_plot_categories')
except Exception as e:
    fail('could not import scripts.classify_and_plot_categories', e)

try:
    import scripts.plot_classify_strategies as pcs
    print('imported scripts.plot_classify_strategies')
except Exception as e:
    fail('could not import scripts.plot_classify_strategies', e)

try:
    import scripts.webviewer_helpers as wvh
    print('imported scripts.webviewer_helpers')
except Exception as e:
    fail('could not import scripts.webviewer_helpers', e)

print('\n-- test compute_output_vector and forward_single_with_nn with two-layer record')
rec_two = {
    'weights': [0.2, -0.3],
    'biases': [0.1, 0.0],
    # second layer: 1 node that reads both layer1 nodes
    'layer2_weights': [[1.0, -0.5]],
    'layer2_biases': [0.2]
}

try:
    outv = g_nn.compute_output_vector(rec_two, [0.5, 0.6])
    print('compute_output_vector(rec_two,[0.5,0.6]) ->', outv)
except Exception as e:
    fail('compute_output_vector failed', e)

try:
    d, m = g_nn.forward_single_with_nn([0.5, 0.6], rec_two)
    print('forward_single_with_nn -> direction, magnitude =', (d, m))
except Exception as e:
    fail('forward_single_with_nn failed', e)

print('\n-- test evaluate_rec from classify_strategies_global')
try:
    vals = csg.evaluate_rec(rec_two, inputs=[-0.5, 0.0, 0.5])
    print('csg.evaluate_rec ->', vals)
except Exception as e:
    fail('csg.evaluate_rec failed', e)

print('\n-- test evaluate_rec from classify_and_plot_categories')
try:
    vals2 = capc.evaluate_rec(rec_two, inputs=[-0.5, 0.0, 0.5])
    print('capc.evaluate_rec ->', vals2)
except Exception as e:
    fail('capc.evaluate_rec failed', e)

print('\nAll import/runtime checks passed')
sys.exit(0)
