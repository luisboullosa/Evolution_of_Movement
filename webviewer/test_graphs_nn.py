import numpy as np
from webviewer import graphs_nn as g


def test_single_input_vectorized():
    # simple network: two nodes with scalar weights and biases
    use_nn = {"weights": [0.5, -0.2], "biases": [0.1, 0.0]}
    xs = np.linspace(-1, 1, 5)
    out = g.compute_nn_response_for(xs, use_nn)
    # naive compute
    expected = np.zeros_like(xs)
    for w, b in zip(use_nn["weights"], use_nn["biases"]):
        expected += np.tanh(w * xs + b)
    assert np.allclose(out, expected)


def test_multi_input_vectorized():
    # network with 2 nodes, each expecting 2 inputs
    use_nn = {"weights": [[1.0, 0.5], [-0.5, 0.2]], "biases": [0.0, 0.1]}
    # build an input_builder that maps center x to [left, center]
    def builder(x):
        return [x - 0.1, x]
    xs = np.linspace(-1, 1, 5)
    out = g.compute_nn_response_for(xs, use_nn, input_builder=builder)
    # compute expected per-x
    expected = []
    for xi in xs:
        inp = builder(float(xi))
        total = 0.0
        for wn, b in zip(use_nn["weights"], use_nn["biases"]):
            dot = wn[0] * inp[0] + wn[1] * inp[1]
            total += np.tanh(dot + b)
        expected.append(total)
    expected = np.array(expected)
    assert np.allclose(out, expected)


if __name__ == "__main__":
    test_single_input_vectorized()
    test_multi_input_vectorized()
    print("tests passed")
