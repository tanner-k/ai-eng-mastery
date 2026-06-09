# Convolutional Neural Networks — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Compute convolution output shape

An input has shape `(B=16, Cin=3, H=32, W=32)`. A convolution has `Cout=64`, kernel size `3x3`, stride `1`, padding `1`, and dilation `1`. What is the output shape?

## Exercise 2 — Count parameters

How many trainable parameters are in the convolution from Exercise 1 if it includes bias? Compare this to a fully connected layer from flattened `32x32x3` input to 64 outputs.

## Exercise 3 — Explain translation equivariance

Explain what it means for convolution to be translation equivariant. Why does weight sharing create this property?

## Exercise 4 — Receptive field growth

For three stacked `3x3` convolutions with stride `1` and padding `1`, what is the receptive field size of one output unit in the final layer? Assume no dilation.

## Exercise 5 — Choose stride or pooling

A CNN for small images loses validation accuracy when an early convolution changes from stride `1` to stride `2`. Give two plausible reasons and two alternatives to reduce compute with less accuracy loss.
