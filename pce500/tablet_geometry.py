"""Shared coordinate arithmetic; model callers supply live ROM coefficients."""


def quarter_to_adc(point, calibration):
    if len(point) != 2 or len(calibration) != 6:
        raise ValueError("tablet coordinate/calibration shape")
    result = []
    for axis, coordinate in enumerate(point):
        denominator, numerator, offset = calibration[axis * 3 : axis * 3 + 3]
        if not 1 <= denominator <= 32767 or not 1 <= numerator <= 32767:
            raise ValueError("tablet calibration is unavailable or unsupported")
        if not -32768 <= coordinate <= 32767 or not -32768 <= offset <= 32767:
            raise ValueError("tablet quarter coordinate/offset outside signed word")
        product = (coordinate - offset) * numerator
        value = abs(product) // denominator * (1 if product >= 0 else -1)
        if not 0 <= value <= 1023:
            raise ValueError("calibrated contact outside ten-bit ADC range")
        result.append(value)
    return tuple(result)
