# Translating MPLang to MPNNs

This repository contains the Python implementation developed for the bachelor's thesis **“Vertaling van MPLang naar MPNN”** by Nette Beckers at Hasselt University. It explores how expressions in MPLang, a language for describing graph feature-map transformations, can be translated into equivalent Message Passing Neural Network (MPNN) layer sequences.

The work builds on the theoretical research on Global Feature Map Transformers (GFMTs) and MPLang by Prof. Dr. Jan Van den Bussche and Juno Steegmans. The thesis and its accompanying poster are included in this repository:

- [Bachelor's thesis](Vertaling%20van%20MPLang%20naar%20MPNN%20Nette%20Beckers%20%282159098%29.pdf)
- [Project poster](Poster%20vertaling%20van%20MPLang%20naar%20MPNN%20Nette%20Beckers%20%282159098%29.pdf)

## Background

An MPNN transforms node feature vectors by applying a sequence of layers. Each layer combines a transformation of a node's own features with a transformation of the sum of its neighbours' features, adds a bias, and applies an activation function. MPLang provides a compact, compositional way to describe such transformations.

This project implements the translation of MPLang expressions into ReLU-MPNNs. Intermediate layers use ReLU activations where needed; the final layer uses the identity activation. The `test.py` example evaluates each expression both directly in MPLang and through its generated MPNN, providing a small check of the translation.

## Supported expressions

The implementation provides these expression forms:

| Expression | Meaning |
| --- | --- |
| `Number(a)` | Constant value `a` at every node |
| `Component(i)` | The `i`th input feature of each node (one-based indexing) |
| `ScalairProduct(e, a)` | Multiply expression `e` by scalar `a` |
| `Sum(e1, e2)` | Add two expressions |
| `Function(f, e)` | Apply an activation function to `e` |
| `Diamond(e)` | Sum the values of `e` over each node's neighbours |

Expressions can be nested to build larger transformations. For example, `Diamond(Sum(Number(5), Component(2)))` adds 5 to each node's second feature, then sums those values over its neighbours.

## Requirements

- Python 3.9 or later
- NumPy

Install NumPy if it is not already available:

```bash
python3 -m pip install numpy
```

## Run the example

From the repository root, run:

```bash
python3 test.py
```

The script creates a graph and several expressions, constructs an MPNN for each expression, then prints the generated network and both evaluation results.

To use the implementation in your own script:

```python
import numpy as np

from Graph.graph import Graph
from MPLang.component import Component
from MPLang.diamond import Diamond
from MPLang.number import Number
from MPLang.sum import Sum

features = np.array([[2, 4], [3, 5]])
graph = Graph([[0, 1], [1, 0]], features)
expression = Diamond(Sum(Number(5), Component(2)))

direct_result = expression.evaluate(features, graph)
mpnn = expression.createMPNN(graph.getNumberOfFeatures())
translated_result = mpnn.evaluate(features, graph)

print(direct_result)
print(translated_result)
```

The two results represent the direct MPLang evaluation and the evaluation of its generated MPNN, respectively.

## Project structure

- `MPLang/`: expression classes and their direct evaluation and MPNN translation.
- `MPNN/`: MPNN layers, evaluation, and network operations such as concatenation.
- `Graph/`: graph representation, adjacency information, and node features.
- `Function/`: activation-function implementation.
- `test.py`: runnable demonstration comparing direct and translated evaluation.

## Scope and assumptions

The graph implementation represents an undirected graph through a symmetric adjacency matrix and the examples use graphs without self-loops. Neighbour aggregation is a sum. This is an educational/research prototype of the translation described in the thesis; it is not a training framework or a production-ready GNN library. The included example checks representative expressions on one graph rather than providing an exhaustive formal verification of all possible inputs.