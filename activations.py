import math


def sigmoid(x: float) -> float:
    """
    The sigmoid function maps any real-valued input number to a value
    between 0 and 1.

    Args:
        x (float):
            The input value before activation.
            x can be any real number, for example -10.0, 0.0, or 10.0.

    Returns:
        float:
            The activated output value.
            The result is always in the range (0, 1).
            Values close to 0 represent a low activation.
            Values close to 1 represent a high activation.
            If x is 0, the result is exactly 0.5.

    Why this formula:
        sigmoid(x) = 1 / (1 + e^(-x))

        We use this formula because it smoothly compresses very small and
        very large input values into the range between 0 and 1.

        - If x is very negative, e^(-x) becomes very large, so the result
          gets close to 0.
        - If x is 0, e^(-x) is 1, so the result is 1 / (1 + 1) = 0.5.
        - If x is very positive, e^(-x) becomes very small, so the result
          gets close to 1.

        This makes sigmoid useful when we want to interpret the output as
        a probability-like value or as an on/off activation strength.

        We do not use a random formula here: this specific shape is smooth,
        bounded, differentiable, and historically useful for neural networks.
    """
    return 1 / (1 + math.exp(-x))


def tanh(x: float) -> float:
    """
    Hyperbolic tangent (tanh) activation function.

    The tanh function maps any real-valued input number to a value
    between -1 and 1.

    Args:
        x (float):
            The input value before activation.
            x can be any real number, for example -10.0, 0.0, or 10.0.

    Returns:
        float:
            The activated output value.
            The result is always in the range (-1, 1).
            Values close to -1 represent a strong negative activation.
            Values close to 1 represent a strong positive activation.
            If x is 0, the result is exactly 0.0.

    Why this formula:
        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

        We use this formula because it smoothly compresses very small and
        very large input values into the range between -1 and 1.

        - If x is very negative, e^(-x) becomes very large, so the result
          gets close to -1.
        - If x is 0, e^x and e^(-x) are both 1, so the result is
          (1 - 1) / (1 + 1) = 0.
        - If x is very positive, e^x becomes very large, so the result
          gets close to 1.

        This makes tanh useful when we want activations that can represent
        both negative and positive signals.

        Compared to sigmoid, tanh is centered around 0 instead of 0.5.
        That often makes it easier for neural networks to learn, because
        negative inputs produce negative outputs and positive inputs produce
        positive outputs.

        We do not use a random formula here: this specific shape is smooth,
        bounded, differentiable, and symmetric around 0.
    """
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def step(x: float) -> int:
    """
    Perceptron step activation function.

    Returns:
        +1 if x is greater than or equal to 0.
        -1 if x is smaller than 0.
    """
    if x >= 0:
        return 1

    return -1


def relu(x: float) -> float:
    """
    Rectified Linear Unit (ReLU) activation function.

    The ReLU function keeps positive input values unchanged and turns
    negative input values into 0.

    Args:
        x (float):
            The input value before activation.
            x can be any real number, for example -10.0, 0.0, or 10.0.

    Returns:
        float:
            The activated output value.
            The result is always in the range [0, infinity).
            Negative inputs return 0.
            Positive inputs return the original input value.
            If x is 0, the result is 0.

    Why this formula:
        relu(x) = max(0, x)

        We use this formula because it is simple, fast, and helps neural
        networks learn non-linear patterns.

        - If x is negative, the output becomes 0.
        - If x is 0, the output stays 0.
        - If x is positive, the output stays x.

        This makes ReLU useful because it does not compress positive values
        like sigmoid does. Large positive signals can stay large, while
        negative signals are blocked.

        ReLU is widely used in hidden layers of neural networks because it
        is computationally cheap and often trains faster than sigmoid or tanh.
    """
    if x > 0:
        return x
    
    return 0

if __name__ == "__main__":
    print(sigmoid(-10.0))
    print(sigmoid(0.0))
    print(sigmoid(100.0))
