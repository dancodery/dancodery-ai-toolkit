"""
FRACTALNET: ULTRA-DEEP NEURAL NETWORKS WITHOUT RESIDUALS

Description:
    - Fractalnets do not include pass-through connections 
    - Fractalnets do not include residual connections

Use Cases:
    - ImageNet
    - CIFAR-10, CIFAR-100

Paper:
    - https://arxiv.org/pdf/1605.07648
"""

def pool(x):
    """Pooling operation."""
    pass

def fractalnet(input_x, block_depth: int) -> None:
    interative_result = input_x

    for i in range(block_depth):
        block_output = fractal_function(4, interative_result)
        interative_result = pool(block_output)

    fractanet_output = prediction(interative_result)

    return fractanet_output


def fractal_function(column_depth: int, input_z) -> None:
    if column_depth <= 1:
        return convolution(input_z)

    left_stem_output = convolution(input_z)

    right_stem_first_fractal_output = fractal_function(column_depth - 1, input_z)
    right_stem_second_fractal_output = fractal_function(column_depth - 1, right_stem_first_fractal_output)
   
    output = join(left_stem_output, right_stem_second_fractal_output)

    return output

if __name__ == "__main__":
    pass