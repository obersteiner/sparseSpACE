import unittest
import sparseSpACE

from sparseSpACE.Extrapolation import ExtrapolationVersion, RombergWeightFactory


class TestWeightFactory(unittest.TestCase):
    def setUp(self) -> None:
        self.places = 8

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Version: Romberg Linear

    def test_extrapolation_coefficient_version_linear(self):
        m = 2

        # Unit domain
        factory = RombergWeightFactory.get(0, 1, version=ExtrapolationVersion.ROMBERG_LINEAR)

        # WolframAlpha: ((a)^1 / ((a)^1 - (b)^1)) * ((c)^1 / ((c)^1 - (b)^1)) for a = 1/2, b = 1, c = 1/4
        self.assertAlmostEqual(1/3, factory.get_extrapolation_coefficient(m, 0), self.places)

        # WolframAlpha: ((a)^1 / ((a)^1 - (b)^1)) * ((c)^1 / ((c)^1 - (b)^1)) for a = 1, b = 1/2, c = 1/4
        self.assertAlmostEqual(-2, factory.get_extrapolation_coefficient(m, 1), self.places)

        # WolframAlpha: ((a)^1 / ((a)^1 - (b)^1)) * ((c)^1 / ((c)^1 - (b)^1)) for a = 1, b = 1/4, c = 1/2
        self.assertAlmostEqual(8/3, factory.get_extrapolation_coefficient(m, 2), self.places)

        # Non-unit domain
        factory = RombergWeightFactory.get(3, 1, version=ExtrapolationVersion.ROMBERG_LINEAR)

        # WolframAlpha: ((a)^1 / ((a)^1 - (b)^1)) * ((c)^1 / ((c)^1 - (b)^1)) for a = (3-1)/2, b = (3-1)/1, c = (3-1)/4
        self.assertAlmostEqual(1/3, factory.get_extrapolation_coefficient(m, 0), self.places)

        # WolframAlpha: ((a)^1 / ((a)^1 - (b)^1)) * ((c)^1 / ((c)^1 - (b)^1)) for a = (3-1)/1, b = (3-1)/2, c = (3-1)/4
        self.assertAlmostEqual(-2, factory.get_extrapolation_coefficient(m, 1), self.places)

        # WolframAlpha: ((a)^1 / ((a)^1 - (b)^1)) * ((c)^1 / ((c)^1 - (b)^1)) for a = (3-1)/1, b = (3-1)/4, c = (3-1)/2
        self.assertAlmostEqual(8/3, factory.get_extrapolation_coefficient(m, 2), self.places)

    def test_weights_version_linear(self):
        a = 0
        b = 1

        factory = RombergWeightFactory.get(a, b, version=ExtrapolationVersion.ROMBERG_LINEAR)
        m = 2

        self.assertAlmostEqual(0, factory.get_boundary_point_weight(m))
        self.assertAlmostEqual(-1/3, factory.get_inner_point_weight(1, m))
        self.assertAlmostEqual(2/3, factory.get_inner_point_weight(2, m))

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Version: Romberg Default

    # Unit domain
    def test_extrapolation_coefficient_version_romberg(self):
        m = 2

        # Unit domain
        factory = RombergWeightFactory.get(0, 1, version=ExtrapolationVersion.ROMBERG_DEFAULT)

        #  WolframAlpha: ((a)^2 / ((a)^2 - (b)^2)) * ((c)^2 / ((c)^2 - (b)^2)) for a = 1/2, b = 1, c = 1/4
        self.assertAlmostEqual(1/45, factory.get_extrapolation_coefficient(m, 0), self.places)

        # WolframAlpha: ((a)^2 / ((a)^2 - (b)^2)) * ((c)^2 / ((c)^2 - (b)^2)) for a = 1, b = 1/2, c = 1/4
        self.assertAlmostEqual(-4/9, factory.get_extrapolation_coefficient(m, 1), self.places)

        #  WolframAlpha: ((a)^2 / ((a)^2 - (b)^2)) * ((c)^2 / ((c)^2 - (b)^2)) for a = 1, b = 1/4, c = 1/2
        self.assertAlmostEqual(64/45, factory.get_extrapolation_coefficient(m, 2), self.places)

        # Non-Unit domain: Same coefficients as in unit domain (factor H out and reduce fraction)
        factory = RombergWeightFactory.get(1, 3, version=ExtrapolationVersion.ROMBERG_DEFAULT)

        # WolframAlpha: ((a)^2 / ((a)^2 - (b)^2)) * ((c)^2 / ((c)^2 - (b)^2))
        #   for a = (3-1) / 2, b = (3-1) / 1, c = (3-1) /4
        self.assertAlmostEqual(1/45, factory.get_extrapolation_coefficient(m, 0), self.places)

        # WolframAlpha: ((a)^2 / ((a)^2 - (b)^2)) * ((c)^2 / ((c)^2 - (b)^2))
        #   for a = (3-1) / 1, b = (3-1) / 2, c = (3-1) /4
        self.assertAlmostEqual(-4/9, factory.get_extrapolation_coefficient(m, 1), self.places)

        # WolframAlpha: ((a)^2 / ((a)^2 - (b)^2)) * ((c)^2 / ((c)^2 - (b)^2))
        #   for a = (3-1) / 1, b = (3-1) / 4, c = (3-1) /2
        self.assertAlmostEqual(64/45, factory.get_extrapolation_coefficient(m, 2), self.places)

    def test_weights_version_romberg(self):
        a = 0
        b = 1

        factory = RombergWeightFactory.get(a, b, version=ExtrapolationVersion.ROMBERG_DEFAULT)
        m = 2

        self.assertAlmostEqual(7/90, factory.get_boundary_point_weight(m))
        self.assertAlmostEqual(2/15, factory.get_inner_point_weight(1, m))
        self.assertAlmostEqual(16/45, factory.get_inner_point_weight(2, m))


if __name__ == '__main__':
    unittest.main()
