# Optimizing Training for Edge-Devices

## Gradient Quantization

Parallelizing SGD comes with a huge communication overhead between nodes. Heuristics based lossy compression techniques communicate quantized gradients between nodes which does not always result in convergence. Alistarh et al. propose QSGD which guarantess convergence by adopting a smooth trade-off between communication bandwidth and convergence time.

---

**_MAIN IDEAS:_**

1. _Intutitive Stochasitc Quantization_: Given a gradient vector at a processor, $g_i$, quantize each component by randomized rounding to a discrete set of values

2. _Efficient LossLess Encoding_: A processor $p_i$ communicates a subset $n_i^j \subset N$ per iteration, $j$, introducing additional variance $\sigma_i^j$ in the entire pipeline.

---

QSGD guarantees convergence and provides tight-bounds on the precision-variance tradeoff and is therefore ideal in terms of stability

\citet{alistarh2017qsgd} proposed Quantized Stochastic Gradient Descent (QSGD) which stands out from other heuristics based quantization techniques for SGD by guaranteeing convergence. The main idea behind QSGD is to adopt a smooth trade-off between gradient precision (for communication bandwidth bottlenecks) and added variance (leading to slower convergence rates). By adopting this strategy, the authors were able to provide tight-bounds on the precision-variance tradeoff thus increasing stability while improving performance

## Gradient Filtering

Gradient Quantization reduces the cost of arithmetic operations but not the number of such operations and therefore limits training speedups. \citet{yang2023efficient}