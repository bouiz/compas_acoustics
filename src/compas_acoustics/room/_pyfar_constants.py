"""Cramer's speed of sound and ISO 9613-1 air attenuation, copied from pyfar 0.8.1 (``pyfar.constants``).

``pyfar.constants`` needs pyfar >= 0.8, which needs Python >= 3.11; Rhino 8 runs Python 3.9. The two
functions are used only there, when ``pyfar.constants`` cannot be imported. They return pyfar objects,
which pyfar 0.7.5 has. ``tests/test_air.py`` compares them with pyfar's wherever both exist.

Copyright (c) [2020] [The pyfar developers]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pyfar as pf

REFERENCE_ATMOSPHERIC_PRESSURE = 101325.0
ABSOLUTE_ZERO_CELSIUS = -273.15
REFERENCE_AIR_TEMPERATURE_CELSIUS = 20


def speed_of_sound_cramer(
        temperature, relative_humidity, co2_ppm=425.19,
        atmospheric_pressure=None,
        ):
    r"""
    Calculate the speed of sound in air based on temperature, atmospheric
    pressure, humidity and CO\ :sub:`2` concentration.

    This implements Cramers method described in [#]_.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius from 0 deg C to 30 deg C.
    relative_humidity : float, array_like
        Relative humidity in the range of 0 to 1.
    co2_ppm : float, array_like, optional
        CO\ :sub:`2` concentration in parts per million. The default is
        425.19 ppm, based on [#]_. Value must be below
        10 000 ppm (1%).
    atmospheric_pressure : float, array_like, optional
        Atmospheric pressure in Pascal, by default
        ``REFERENCE_ATMOSPHERIC_PRESSURE``.
        Value must be between 75 000 Pa to 102000 Pa.

    Returns
    -------
    speed_of_sound : float, array_like
        Speed of sound in air in m/s.

    References
    ----------
    .. [#] O. Cramer, "The variation of the specific heat ratio and the
           speed of sound in air with temperature, pressure, humidity, and
           CO\ :sub:`2` concentration," The Journal of the Acoustical Society
           of America, vol. 93, no. 5, pp. 2510-2516, May 1993,
           doi: 10.1121/1.405827.
    .. [#] Lan, X., Tans, P. and K.W. Thoning: Trends in globally-averaged
           CO2 determined from NOAA Global Monitoring Laboratory measurements.
           Version Friday, 14-Mar-2025 11:33:44 MDT
           https://doi.org/10.15138/9N0H-ZH07

    """
    if atmospheric_pressure is None:
        atmospheric_pressure = REFERENCE_ATMOSPHERIC_PRESSURE
    temperature = np.array(temperature)
    relative_humidity = np.array(relative_humidity)
    atmospheric_pressure = np.array(atmospheric_pressure)
    co2_ppm = np.array(co2_ppm)

    if np.any(temperature < 0) or np.any(temperature > 30):
        raise ValueError("Temperature must be between 0°C and 30°C.")
    if np.any(relative_humidity < 0) or np.any(relative_humidity > 1):
        raise ValueError("Relative humidity must be between 0 and 1.")
    if np.any(atmospheric_pressure < 75e3) or np.any(
            atmospheric_pressure > 102e3):
        raise ValueError(
            "Atmospheric pressure must be between 75 000 Pa to 102 000 Pa.")
    if np.any(co2_ppm < 0) or np.any(co2_ppm > .01e6):
        raise ValueError(
            "CO2 concentration (ppm) must be between 0 ppm to 10 000 ppm.")

    # mole fraction of CO2 in air
    x_c = co2_ppm * 1e-6
    # pressure(atmospheric pressure) in Pa
    p = atmospheric_pressure
    # relative humidity as a fraction
    h = relative_humidity
    # temperature in Celsius
    t = temperature
    # thermodynamic temperature
    T = t - ABSOLUTE_ZERO_CELSIUS

    # enhancement factor f (Equation A2)
    f = 1.00062+3.14e-8 * p + 5.6e-7*temperature**2
    # saturation vapor pressure in air (Equation A3)
    p_sv = np.exp(1.2811805e-5*T**2-1.9509874e-2*T+34.04926034-6.3536311e3/T)
    # water vapor mole fraction (Equation A1)
    x_w = h * f * p_sv / p

    # check water mole fraction after Cramer
    if np.any(x_w < 0) or np.any(x_w > 0.06):
        raise ValueError("Water mole fraction must be between 0 and 0.06.")

    # Coefficients according to Cramer (Table. III)
    a0 = 331.5024
    a1 = 0.603055
    a2 = -0.000528
    a3 = 51.471935
    a4 = 0.1495874
    a5 = -0.000782
    a6 = -1.82e-7
    a7 = 3.73e-8
    a8 = -2.93e-10
    a9 = -85.20931
    a10 = -0.228525
    a11 = 5.91e-5
    a12 = -2.835149
    a13 = -2.15e-13
    a14 = 29.179762
    a15 = 0.000486

    # approximation for c according to Cramer (Eq. 15)
    c1 = a0+a1*t+a2*t**2
    c2 = (a3+a4*t+a5*t**2)*x_w
    c3 = (a6+a7*t+a8*t**2)*p
    c4 = (a9+a10*t+a11*t**2)*x_c
    c5 = a12*x_w**2 + a13*p**2 + a14*x_c**2 + a15*x_w*p*x_c

    return c1 + c2 + c3 + c4 + c5


def air_attenuation(
        temperature, frequencies, relative_humidity,
        atmospheric_pressure=None):
    r"""Calculate the pure tone attenuation of sound in air according to
    ISO 9613-1.

    Calculation is in accordance with ISO 9613-1 [#]_. The shape of the
    outputs is broadcasted from the shapes of the ``temperature``,
    ``relative_humidity``, and ``atmospheric_pressure``.
    The frequency bins represents the last dimension.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in deg C.
        Must be in the range of -20 deg C to 50 deg C for accuracy of +/-10% or
        must be greater than -70 deg C for accuracy of +/-50%.
    frequencies : float, array_like
        Frequency in Hz. Must be greater than 50 Hz.
        Just one dimensional array is allowed.
    relative_humidity : float, array_like
        Relative humidity in the range from 0 to 1.
    atmospheric_pressure : float, array_like, optional
        Atmospheric pressure in Pascal, by default
        ``REFERENCE_ATMOSPHERIC_PRESSURE``.

    Returns
    -------
    alpha : np.ndarray[float]
        Pure tone air attenuation coefficient in dB/m for atmospheric
        absorption.
    m : :py:class:`~pyfar.FrequencyData`
        Pure tone energy attenuation coefficient in 1/m for atmospheric
        absorption. The parameter ``m`` is calculated as
        :math:`m = \alpha / (10 \log_{10}(e))`.
    accuracy : :py:class:`~pyfar.FrequencyData`
        accuracy of the results according to the standard:

        ``10``, +/- 10% accuracy
            - molar concentration of water vapour: 0.05% to 5%.
            - air temperature: 253.15 K to 323.15 K (-20 deg C to +50 deg C)
            - atmospheric pressure: less than 200 000 Pa (2 atm)
            - frequency-to-pressure ratio: 0.0004 Hz/Pa to 10 Hz/Pa.

        ``20``, +/- 20% accuracy
            - molar concentration of water vapour: 0.005% to 0.05%,
              and greater than 5%
            - air temperature: 253.15 K to 323.15 K (-20 deg C to +50 deg C)
            - atmospheric pressure: less than 200 000 Pa (2 atm)
            - frequency-to-pressure ratio: 0.0004 Hz/Pa to 10 Hz/Pa.

        ``50``, +/- 50% accuracy
            - molar concentration of water vapour: less than 0.005%
            - air temperature: greater than 200 K (- 73 deg C)
            - atmospheric pressure: less than 200 000 Pa (2 atm)
            - frequency-to-pressure ratio: 0.0004 Hz/Pa to 10 Hz/Pa.

        ``-1``, no valid result
            else.

    References
    ----------
    .. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
           propagation outdoors -- Part 1: Calculation of the absorption of
           sound by the atmosphere.
    """
    if atmospheric_pressure is None:
        atmospheric_pressure = REFERENCE_ATMOSPHERIC_PRESSURE
    if not isinstance(temperature, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'temperature must be a number or array of numbers')
    if not isinstance(frequencies, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'frequencies must be a number or array of numbers')
    if not isinstance(
            relative_humidity, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'relative_humidity must be a number or array of numbers')
    if np.array(frequencies).ndim > 1:
        raise ValueError('frequencies must be one dimensional.')
    if not isinstance(
            atmospheric_pressure, (int, float, np.ndarray, list, tuple)):
        raise TypeError(
            'atmospheric_pressure must be a number or array of numbers')

    try:
        _ = np.broadcast_shapes(
            np.atleast_1d(temperature).shape,
            np.atleast_1d(relative_humidity).shape,
            np.atleast_1d(atmospheric_pressure).shape)
    except ValueError as e:
        raise ValueError(
            'temperature, relative_humidity, and atmospheric_pressure must '
            'have the same shape or be broadcastable.') from e

    if np.any(np.array(temperature) < -73):
        raise ValueError("Temperature must be greater than -73°C.")
    if np.any(np.array(frequencies) < 50):
        raise ValueError("frequencies must be greater than 50 Hz.")
    if np.any(np.array(relative_humidity) < 0) or np.any(
            np.array(relative_humidity) > 1):
        raise ValueError("Relative humidity must be between 0 and 1.")
    if np.any(np.array(atmospheric_pressure) > 200000):
        raise ValueError("Atmospheric pressure must be less than 200 kPa.")
    temperature = np.array(
        temperature, dtype=float)[..., np.newaxis]
    relative_humidity = np.array(
        relative_humidity, dtype=float)[..., np.newaxis]
    atmospheric_pressure = np.array(
        atmospheric_pressure, dtype=float)[..., np.newaxis]
    frequencies = np.array(frequencies, dtype=float)

    p_atmospheric_ref = REFERENCE_ATMOSPHERIC_PRESSURE
    t_degree_ref = REFERENCE_AIR_TEMPERATURE_CELSIUS

    h_r = relative_humidity*100
    p_a = atmospheric_pressure
    p_r = p_atmospheric_ref
    f = frequencies
    T = temperature - ABSOLUTE_ZERO_CELSIUS
    T_0 = t_degree_ref - ABSOLUTE_ZERO_CELSIUS

    p_sat = _saturation_vapour_pressure_iso(temperature)

    # molar concentration of water vapor as a percentage (Equation B.1)
    h = h_r * (p_sat / p_r) * (p_a / p_r)

    # Oxygen relaxation frequency (Eq. 3)
    f_rO = (p_a/p_r)*(24+4.04e4*h*(0.02+h)/(0.391+h))

    # Nitrogen relaxation frequency (Eq. 4)
    f_rN = (p_a/p_r)*(T/T_0)**(-1/2)*(9+280*h*np.exp(
        -4.17*((T/T_0)**(-1/3)-1)))

    # air attenuation (Eq. 5)
    alpha = 8.686*f**2*((1.84e-11*p_r/p_a*(T/T_0)**(1/2)) + \
        (T/T_0)**(-5/2)*(0.01275*np.exp(-2239.1/T)*(f_rO + (f**2/f_rO))**(-1)
        +0.1068*np.exp(-3352/T) * (f_rN + (f**2/f_rN))**(-1)))

    # Equation 3: ISO 17497-1:2004
    m = pf.FrequencyData(
        alpha / (10*np.log10(np.exp(1))), frequencies=frequencies)

    accuracy = _air_attenuation_accuracy(
        h, temperature, atmospheric_pressure, frequencies, m.freq.shape)

    return alpha, m, accuracy


def _air_attenuation_accuracy(
        concentration_water_vapour, temperature, atmospheric_pressure,
        frequencies, shape):
    """Calculate the accuracy of the air attenuation calculation.

    Calculation is in accordance with ISO 9613-1 [#]_. This method is used in
    :py:func:`air_attenuation` to calculate the accuracy of
    the air attenuation calculation.

    Parameters
    ----------
    concentration_water_vapour : float, array_like
        Molar concentration of water vapor as a percentage.
        Must be between 0% and 100%.
    temperature : float, array_like
        Temperature in degree Celsius.
        Must be above -273.15 deg C.
    atmospheric_pressure : float, array_like
        Atmospheric pressure in Pascal.
        Must be above 0Pa.
    frequencies : float, array_like
        Frequency in Hz.
        Must be larger than 0 Hz.
    shape : tuple
        Shape of the output.

    Returns
    -------
    accuracy : :py:class:`~pyfar.classes.FrequencyData`
        accuracy of the results according to the standard:

            ``10``, +/- 10% accuracy
                - molar concentration of water vapour: 0.05% to 5%.
                - air temperature: 253.15 K to 323.15 K (-20 deg C to +50 deg C)
                - atmospheric pressure: less than 200 000 Pa (2 atm)
                - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

            ``20``, +/- 20% accuracy
                - molar concentration of water vapour: 0.005% to 0.05%,
                  and greater than 5%
                - air temperature: 253.15 K to 323.15 K (-20 deg C to +50 deg C)
                - atmospheric pressure: less than 200 000 Pa (2 atm)
                - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

            ``50``, +/- 50% accuracy
                - molar concentration of water vapour: less than 0.005%
                - air temperature: greater than 200 K (- 73 deg C)
                - atmospheric pressure: less than 200 000 Pa (2 atm)
                - frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa.

            ``-1``, no valid result
                else.

    References
    ----------
    .. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
           propagation outdoors -- Part 1: Calculation of the absorption of
           sound by the atmosphere.
    """
    if np.any(np.array(concentration_water_vapour) < 0) or np.any(
            np.array(concentration_water_vapour) > 100):
        raise ValueError(
            r"Concentration of water vapour must be between 0% and 100%.")
    if np.any(np.array(temperature) < -273.15):
        raise ValueError(
            "Temperature must be greater than -273.15°C.")
    if np.any(np.array(atmospheric_pressure) < 0):
        raise ValueError(
            "Atmospheric pressure must be greater than 0 Pa.")
    if np.any(np.array(frequencies) < 0):
        raise ValueError(
            "Frequencies must be positive.")

    atmospheric_pressure = np.broadcast_to(atmospheric_pressure, shape)
    h_water_vapor = np.broadcast_to(concentration_water_vapour, shape)
    frequency_pressure_ratio = frequencies/atmospheric_pressure
    accuracy = np.zeros(shape) - 1

    # atmospheric pressure < 200 kPa
    atm_mask = atmospheric_pressure < 200000
    # frequency-to-pressure ratio: 4 x 10-4 Hz/Pa to 10 Hz/Pa
    frequency_pressure_ratio_mask = (4e-4 <= frequency_pressure_ratio) & (
        frequency_pressure_ratio <= 10)
    common_mask = atm_mask & frequency_pressure_ratio_mask

    # molar concentration of water vapour: 0.05% to 5%
    vapor_10_mask = (0.05 <= h_water_vapor) & (h_water_vapor <= 5)
    # molar concentration of water vapour: 0.005% to 0.05% and greater than 5%
    vapor_20_mask = (5 < h_water_vapor) | (
        (0.005 <= h_water_vapor) & (h_water_vapor < 0.05))
    # molar concentration of water vapour: less than 0.005%
    vapor_50_mask = (0.005 > h_water_vapor)

    # air temperature: 253,15 K to 323,15 (-20 deg C to +50 deg C)
    temp_20_mask = (-20 <= temperature) & (temperature <= 50)
    # air temperature: greater than 200 K (- 73 deg C)
    temp_50_mask = (-73 <= temperature)

    accuracy_50 = common_mask & temp_50_mask & (
        vapor_10_mask | vapor_20_mask | vapor_50_mask)
    accuracy[accuracy_50] = 50

    accuracy_20 = common_mask & temp_20_mask & (
        vapor_10_mask | vapor_20_mask)
    accuracy[accuracy_20] = 20

    accuracy_10 = vapor_10_mask & common_mask & temp_20_mask
    accuracy[accuracy_10] = 10

    return pf.FrequencyData(accuracy, frequencies=frequencies)


def _saturation_vapour_pressure_iso(temperature):
    """Calculates the saturation vapour pressure after ISO 9613-1:1993.

    This method is used in the :py:func:`air_attenuation` function to calculate
    the saturation vapour pressure of water vapour in air as a close
    approximation to
    those calculated by the World Meteorological Organization.
    The method is described in Equation B.2 and B.3 in [#]_.

    Parameters
    ----------
    temperature : float, array_like
        Temperature in degree Celsius.
        Must be in the range of -20 deg C to 50 deg C for accuracy of +/-10% or
        must be greater than -70 deg C for accuracy of +/-50%.

    Returns
    -------
    p_sat : float, array_like
        Saturation vapour pressure in Pascal.

    References
    ----------
    .. [#] ISO 9613-1:1993, Acoustics -- Attenuation of sound during
           propagation outdoors -- Part 1: Calculation of the absorption of
           sound by the atmosphere.
    """
    T = temperature - ABSOLUTE_ZERO_CELSIUS
    p_r = REFERENCE_ATMOSPHERIC_PRESSURE
    # triple-point isotherm temperature of 273.16 K
    T_01 = 273.16

    # saturation vapour pressure (Equation B.2 and B.3)
    C = -6.8346*(T_01/T)**1.261+4.6151
    return 10**C * p_r
