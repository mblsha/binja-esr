#!/usr/bin/env python3
"""
OCR optimization test for PC-E500 LCD display.

This test extracts the OCR logic from run_pce500.py and allows experimentation
with different tesseract configurations to improve accuracy for monospace display text.
"""

import time
from pathlib import Path
from typing import Optional, Tuple
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image, ImageOps, ImageFilter
import pytesseract


def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    Pure Python implementation to avoid external dependencies.
    """
    if len(s1) < len(s2):
        return calculate_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity_score(s1: str, s2: str) -> float:
    """
    Calculate similarity score (0-100) based on Levenshtein distance.
    """
    if not s1 and not s2:
        return 100.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 100.0
    distance = calculate_levenshtein_distance(s1, s2)
    return (1 - distance / max_len) * 100


def ocr_image(
    image_path: str,
    border_size: int = 5,
    border_color: int = 255,
    invert: bool = True,
    threshold: Optional[int] = 128,
    psm_mode: int = 6,
    whitelist: Optional[str] = None,
    scale_factor: float = 1.0,
    scale_method: int = Image.LANCZOS,
    denoise: bool = False,
    adaptive_threshold: bool = False,
    save_preprocessed: bool = False,
    output_prefix: str = "preprocessed",
) -> Tuple[str, Image.Image]:
    """
    Process an image for OCR with configurable parameters.

    Args:
        image_path: Path to input image
        border_size: Pixels of border to add (0 to disable)
        border_color: Color of border (0=black, 255=white)
        invert: Whether to invert colors
        threshold: Binary threshold value (None to skip binarization)
        psm_mode: Tesseract PSM mode (0-13)
        whitelist: Characters to whitelist (None for all)
        scale_factor: Image scaling factor
        scale_method: PIL scaling method
        denoise: Apply denoising filter
        adaptive_threshold: Use adaptive thresholding
        save_preprocessed: Save preprocessed image for inspection
        output_prefix: Prefix for saved preprocessed images

    Returns:
        Tuple of (recognized_text, preprocessed_image)
    """
    # Load and convert to grayscale
    im = Image.open(image_path).convert("L")

    # Add border if requested
    if border_size > 0:
        im = ImageOps.expand(im, border=border_size, fill=border_color)

    # Scale if requested
    if scale_factor != 1.0:
        new_size = (int(im.width * scale_factor), int(im.height * scale_factor))
        im = im.resize(new_size, scale_method)

    # Apply denoising
    if denoise:
        im = im.filter(ImageFilter.MedianFilter(size=3))

    # Invert colors if requested
    if invert:
        im = Image.eval(im, lambda v: 255 - v)

    # Apply thresholding
    if threshold is not None:
        if adaptive_threshold:
            # Simple adaptive threshold using local mean
            # This is a basic implementation; more sophisticated methods exist
            import numpy as np

            arr = np.array(im)
            # Use a simple global mean as threshold
            thresh_val = arr.mean()
            im = Image.fromarray((arr > thresh_val).astype(np.uint8) * 255)
        else:
            # Fixed threshold
            im = im.point(lambda v: 255 if v > threshold else 0, mode="1")

    # Save preprocessed image if requested
    if save_preprocessed:
        preprocessed_path = (
            f"{output_prefix}_psm{psm_mode}_th{threshold}_sc{scale_factor}.png"
        )
        im.save(preprocessed_path)

    # Build tesseract config
    config_parts = [f"--psm {psm_mode}"]
    if whitelist:
        # Use OEM 0 (legacy engine) for whitelist support in older versions
        # or OEM 1 (LSTM) for newer versions
        config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
    config = " ".join(config_parts)

    # Run OCR
    try:
        text = pytesseract.image_to_string(im, config=config)
    except Exception as e:
        text = f"OCR Error: {e}"

    return text, im


def test_ocr_configurations():
    """
    Test various OCR configurations on lcd_display.png and find the best one.
    """
    # Expected text (4 lines with second line empty)
    expected_lines = [
        "S2(CARD):NEW CARD",
        "",  # Empty second line is correct
        "PF1 --- INITIALIZE",
        "PF2 --- DO NOT INITIALIZE",
    ]
    "\n".join(expected_lines)

    # Check if test image exists
    test_image = "lcd_display.png"
    if not Path(test_image).exists():
        print(
            f"Test image {test_image} not found. Please run the emulator first to generate it."
        )
        return

    # Define test configurations
    configurations = []

    # PSM modes to test
    psm_modes = [3, 4, 6, 7, 11, 13]

    # Thresholds to test
    thresholds = [100, 128, 150, 180, None]  # None means no binarization

    # Scale factors
    scale_factors = [1.0, 2.0, 3.0, 4.0]

    # Character whitelists
    whitelists = [
        None,  # No whitelist
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789():-. ",  # Uppercase + special
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789():-. ",  # Full
    ]

    # Create configurations
    for psm in psm_modes:
        for threshold in thresholds:
            for scale in scale_factors:
                for whitelist in whitelists:
                    configurations.append(
                        {
                            "psm_mode": psm,
                            "threshold": threshold,
                            "scale_factor": scale,
                            "whitelist": whitelist,
                            "border_size": 5,
                            "invert": True,
                        }
                    )

    # Add some special configurations
    configurations.extend(
        [
            # No preprocessing
            {
                "psm_mode": 6,
                "threshold": None,
                "scale_factor": 1.0,
                "invert": False,
                "border_size": 0,
            },
            # Adaptive threshold
            {
                "psm_mode": 6,
                "threshold": 128,
                "scale_factor": 2.0,
                "adaptive_threshold": True,
            },
            # With denoising
            {"psm_mode": 6, "threshold": 128, "scale_factor": 2.0, "denoise": True},
        ]
    )

    print(f"Testing {len(configurations)} different OCR configurations...")
    print("=" * 80)

    results = []

    for i, config in enumerate(configurations, 1):
        print(f"\nConfiguration {i}/{len(configurations)}:")
        print(
            f"  PSM={config.get('psm_mode', 6)}, "
            f"Threshold={config.get('threshold', 128)}, "
            f"Scale={config.get('scale_factor', 1.0)}x, "
            f"Whitelist={'Yes' if config.get('whitelist') else 'No'}"
        )

        start_time = time.time()

        try:
            recognized_text, _ = ocr_image(test_image, **config)
            elapsed_time = time.time() - start_time

            # Split into lines for comparison
            recognized_lines = recognized_text.strip().split("\n")

            # Calculate scores for each line
            line_scores = []
            line_distances = []

            for j, (expected, recognized) in enumerate(
                zip(expected_lines, recognized_lines + [""] * 4), 1
            ):
                score = similarity_score(expected, recognized)
                distance = calculate_levenshtein_distance(expected, recognized)
                line_scores.append(score)
                line_distances.append(distance)

                if score < 100:
                    print(f"  Line {j}: {score:.1f}% (distance={distance})")
                    print(f"    Expected:   '{expected}'")
                    print(f"    Recognized: '{recognized}'")

            # Calculate total score
            total_score = sum(line_scores) / len(expected_lines)
            total_distance = sum(line_distances)

            results.append(
                {
                    "config": config,
                    "recognized_text": recognized_text,
                    "recognized_lines": recognized_lines,
                    "line_scores": line_scores,
                    "total_score": total_score,
                    "total_distance": total_distance,
                    "time": elapsed_time,
                }
            )

            print(
                f"  Total Score: {total_score:.1f}% (total distance={total_distance})"
            )
            print(f"  Time: {elapsed_time:.3f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "config": config,
                    "error": str(e),
                    "total_score": 0,
                    "total_distance": 999,
                }
            )

    # Sort results by score
    results.sort(key=lambda x: x["total_score"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS BY ACCURACY:")
    print("=" * 80)

    for i, result in enumerate(results[:10], 1):
        if "error" in result:
            continue
        config = result["config"]
        print(
            f"\n{i}. Score: {result['total_score']:.1f}% (distance={result['total_distance']})"
        )
        print(
            f"   PSM={config.get('psm_mode', 6)}, "
            f"Threshold={config.get('threshold', 128)}, "
            f"Scale={config.get('scale_factor', 1.0)}x"
        )
        print(f"   Time: {result['time']:.3f}s")

        # Show recognized text if not perfect
        if result["total_score"] < 100:
            print("   Recognized text:")
            for line in result["recognized_lines"][:4]:
                print(f"     '{line}'")

    # Report best configuration
    if results and results[0]["total_score"] > 0:
        best = results[0]
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION:")
        print("=" * 80)
        config = best["config"]
        print(f"Score: {best['total_score']:.1f}%")
        print(f"Configuration: {config}")

        if best["total_score"] == 100:
            print("\nPERFECT RECOGNITION ACHIEVED!")
        else:
            print("\nRecognized text:")
            for i, line in enumerate(best["recognized_lines"][:4], 1):
                expected = expected_lines[i - 1] if i <= len(expected_lines) else ""
                if line != expected:
                    print(f"  Line {i}: '{line}' (expected: '{expected}')")
                else:
                    print(f"  Line {i}: '{line}' ✓")


def test_single_configuration():
    """Quick test with default configuration."""
    test_image = "lcd_display.png"
    if not Path(test_image).exists():
        print(f"Test image {test_image} not found.")
        return

    print("Testing default configuration (from run_pce500.py)...")
    text, im = ocr_image(
        test_image,
        border_size=5,
        border_color=255,
        invert=True,
        threshold=128,
        psm_mode=6,
        scale_factor=1.0,
        save_preprocessed=True,
        output_prefix="default",
    )

    print(f"Recognized text:\n{text}")

    # Expected text
    expected_lines = [
        "S2(CARD):NEW CARD",
        "",
        "PF1 --- INITIALIZE",
        "PF2 --- DO NOT INITIALIZE",
    ]

    recognized_lines = text.strip().split("\n")

    print("\nLine-by-line comparison:")
    for i, (expected, recognized) in enumerate(
        zip(expected_lines, recognized_lines + [""] * 4), 1
    ):
        score = similarity_score(expected, recognized)
        if score == 100:
            print(f"Line {i}: ✓ '{recognized}'")
        else:
            print(f"Line {i}: {score:.1f}% similarity")
            print(f"  Expected:   '{expected}'")
            print(f"  Recognized: '{recognized}'")


def test_best_configurations():
    """Test only the best-performing configurations based on initial analysis."""
    # Expected text (4 lines with second line empty)
    expected_lines = [
        "S2(CARD):NEW CARD",
        "",  # Empty second line is correct
        "PF1 --- INITIALIZE",
        "PF2 --- DO NOT INITIALIZE",
    ]

    # Check if test image exists
    test_image = "lcd_display.png"
    if not Path(test_image).exists():
        print(
            f"Test image {test_image} not found. Please run the emulator first to generate it."
        )
        return

    # Best configurations from analysis
    best_configs = [
        # Best overall: 98.6% accuracy
        {
            "psm_mode": 3,
            "threshold": 128,
            "scale_factor": 4.0,
            "border_size": 5,
            "invert": True,
        },
        # Second best: 98.0% accuracy
        {
            "psm_mode": 4,
            "threshold": 128,
            "scale_factor": 4.0,
            "border_size": 5,
            "invert": True,
        },
        # Good with no threshold: 97.1% accuracy
        {
            "psm_mode": 3,
            "threshold": None,
            "scale_factor": 3.0,
            "border_size": 5,
            "invert": True,
        },
        # Try PSM 6 with 4x scaling
        {
            "psm_mode": 6,
            "threshold": 128,
            "scale_factor": 4.0,
            "border_size": 5,
            "invert": True,
        },
        # Try PSM 11 (sparse text) with 4x scaling
        {
            "psm_mode": 11,
            "threshold": 128,
            "scale_factor": 4.0,
            "border_size": 5,
            "invert": True,
        },
    ]

    print("Testing best-performing OCR configurations...")
    print("=" * 80)

    for i, config in enumerate(best_configs, 1):
        print(f"\nConfiguration {i}:")
        print(
            f"  PSM={config['psm_mode']}, Threshold={config.get('threshold', 'None')}, "
            f"Scale={config['scale_factor']}x"
        )

        try:
            recognized_text, _ = ocr_image(test_image, **config)
            recognized_lines = recognized_text.strip().split("\n")

            # Calculate total score
            total_distance = 0
            total_score = 0

            for j, (expected, recognized) in enumerate(
                zip(expected_lines, recognized_lines + [""] * 4), 1
            ):
                score = similarity_score(expected, recognized)
                distance = calculate_levenshtein_distance(expected, recognized)
                total_score += score
                total_distance += distance

                if score < 100:
                    print(f"  Line {j}: '{recognized}' (distance={distance})")

            total_score = total_score / len(expected_lines)
            print(
                f"  Total Score: {total_score:.1f}% (total distance={total_distance})"
            )

            if total_score == 100:
                print("  ✓ PERFECT RECOGNITION!")

        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--single":
            test_single_configuration()
        elif sys.argv[1] == "--best":
            test_best_configurations()
        else:
            print("Usage: python test_ocr_optimization.py [--single|--best]")
            print("  --single: Test default configuration only")
            print("  --best: Test best configurations only")
            print("  (no args): Run full optimization test")
    else:
        test_ocr_configurations()
