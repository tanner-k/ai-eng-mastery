# Convolutional Neural Networks — Solutions

## Solution 1 — Compute convolution output shape

With padding `1`, kernel `3`, dilation `1`, and stride `1`:

```text
Hout = floor((32 + 2*1 - 1*(3 - 1) - 1) / 1 + 1) = 32
Wout = 32
```

The output shape is `(16, 64, 32, 32)`.

## Solution 2 — Count parameters

Convolution weights: `64 * 3 * 3 * 3 = 1,728`. Biases: `64`. Total: `1,792`.

Flattened fully connected weights: `(32 * 32 * 3) * 64 = 196,608`. Biases: `64`. Total: `196,672`. The convolution uses far fewer parameters because weights are shared spatially.

## Solution 3 — Explain translation equivariance

Translation equivariance means shifting the input shifts the output feature map in the same way. Weight sharing creates this property because the same kernel is applied at every spatial position, so a pattern produces the same response wherever it appears, aside from boundary effects.

## Solution 4 — Receptive field growth

Each stride-1 `3x3` convolution adds two pixels to the receptive field per spatial axis. Starting from `1`, three layers give `1 + 3 * 2 = 7`. The final receptive field is `7x7`.

## Solution 5 — Choose stride or pooling

Early stride `2` may discard fine spatial detail or make small objects vanish. It can also reduce feature-map resolution before the model has learned enough low-level features. Alternatives include delaying downsampling, using stride `1` followed by pooling later, using fewer channels in early layers, depthwise separable convolutions, or reducing input resolution only after validating the accuracy tradeoff.
