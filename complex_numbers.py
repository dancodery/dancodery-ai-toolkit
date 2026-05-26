import math


class ComplexNumber:
    """
    A small learning implementation of complex numbers.

    A complex number has two parts:
        real:
            The normal number part.
        imaginary:
            The coefficient in front of i, where i^2 = -1.

    Example:
        ComplexNumber(3, 2) represents 3 + 2i.
    """

    def __init__(self, real: float, imaginary: float) -> None:
        self.real = real
        self.imaginary = imaginary

    def __repr__(self) -> str:
        if self.imaginary < 0:
            return f"{self.real} - {abs(self.imaginary)}i"

        return f"{self.real} + {self.imaginary}i"

    def __add__(self, other: "ComplexNumber") -> "ComplexNumber":
        """
        Add two complex numbers.

        Formula:
            (a + bi) + (c + di) = (a + c) + (b + d)i
        """
        return ComplexNumber(
            self.real + other.real,
            self.imaginary + other.imaginary,
        )

    def __sub__(self, other: "ComplexNumber") -> "ComplexNumber":
        """
        Subtract two complex numbers.

        Formula:
            (a + bi) - (c + di) = (a - c) + (b - d)i
        """
        return ComplexNumber(
            self.real - other.real,
            self.imaginary - other.imaginary,
        )

    def __mul__(self, other: "ComplexNumber") -> "ComplexNumber":
        """
        Multiply two complex numbers.

        Formula:
            (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        """
        real_part = (self.real * other.real) - (self.imaginary * other.imaginary)
        imaginary_part = (self.real * other.imaginary) + (self.imaginary * other.real)

        return ComplexNumber(real_part, imaginary_part)

    def magnitude(self) -> float:
        """
        Return the distance from 0 + 0i.

        Formula:
            |a + bi| = sqrt(a^2 + b^2)
        """
        return math.sqrt((self.real**2) + (self.imaginary**2))

    def conjugate(self) -> "ComplexNumber":
        """
        Return the complex conjugate of this complex number.

        The complex conjugate keeps the real part unchanged and flips the
        sign of the imaginary part.

        Example:
            3 + 4i becomes 3 - 4i
            3 - 4i becomes 3 + 4i

        Formula:
            conjugate(a + bi) = a - bi

        Returns:
            ComplexNumber:
                A new complex number with the same real part and the opposite
                imaginary part.
        """
        return ComplexNumber(self.real, -self.imaginary)


if __name__ == "__main__":
    first_number = ComplexNumber(1, 2)
    second_number = ComplexNumber(3, 4)
    magnitude_example = ComplexNumber(3, 4)

    print(first_number + second_number)
    print(first_number * second_number)
    print(magnitude_example.magnitude())
    print(magnitude_example.conjugate())
