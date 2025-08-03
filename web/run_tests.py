#!/usr/bin/env python3
"""Test runner for PC-E500 web emulator."""

import sys
import os
import unittest
import time
from pathlib import Path
from io import StringIO

# Set environment variable for mock Binary Ninja API
os.environ['FORCE_BINJA_MOCK'] = '1'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test modules to run
TEST_MODULES = [
    'tests.test_keyboard_handler',
    'tests.test_keyboard_memory_overlay', 
    'tests.test_api_integration',
    'tests.test_emulator_state_updates',
    'tests.test_keyboard_cpu_integration'
]


class ColoredTextTestResult(unittest.TextTestResult):
    """Test result class with colored output."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_count = 0
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.showAll:
            self.stream.writeln("\033[92m✓ PASS\033[0m")
        elif self.dots:
            self.stream.write("\033[92m.\033[0m")
            self.stream.flush()
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.writeln("\033[91m✗ ERROR\033[0m")
        elif self.dots:
            self.stream.write("\033[91mE\033[0m")
            self.stream.flush()
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.writeln("\033[91m✗ FAIL\033[0m")
        elif self.dots:
            self.stream.write("\033[91mF\033[0m")
            self.stream.flush()
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.writeln(f"\033[93m- SKIP: {reason}\033[0m")
        elif self.dots:
            self.stream.write("\033[93ms\033[0m")
            self.stream.flush()


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Test runner with colored output."""
    resultclass = ColoredTextTestResult


def run_tests(verbosity=2):
    """Run all tests and display results."""
    print("PC-E500 Web Emulator Test Suite")
    print("=" * 60)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Load tests from each module
    for module_name in TEST_MODULES:
        try:
            module = __import__(module_name, fromlist=[''])
            module_tests = loader.loadTestsFromModule(module)
            suite.addTests(module_tests)
            test_count = module_tests.countTestCases()
            print(f"Loaded {test_count} tests from {module_name}")
        except Exception as e:
            print(f"\033[91mError loading {module_name}: {e}\033[0m")
    
    print()
    print(f"Total tests to run: {suite.countTestCases()}")
    print("-" * 60)
    print()
    
    # Run tests
    runner = ColoredTextTestRunner(verbosity=verbosity, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print()
    print("=" * 60)
    print("Test Summary")
    print("-" * 60)
    
    total_tests = result.testsRun
    passed = result.success_count
    failed = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    
    print(f"Total tests run: {total_tests}")
    print(f"\033[92mPassed: {passed}\033[0m")
    if failed > 0:
        print(f"\033[91mFailed: {failed}\033[0m")
    if errors > 0:
        print(f"\033[91mErrors: {errors}\033[0m")
    if skipped > 0:
        print(f"\033[93mSkipped: {skipped}\033[0m")
    
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
    
    # Overall result
    print()
    if result.wasSuccessful():
        print("\033[92m✓ All tests passed!\033[0m")
        return 0
    else:
        print("\033[91m✗ Some tests failed.\033[0m")
        return 1


def run_single_test(test_name):
    """Run a single test by name."""
    print(f"Running single test: {test_name}")
    print("-" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    runner = ColoredTextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


def list_tests():
    """List all available tests."""
    print("Available tests:")
    print("-" * 60)
    
    loader = unittest.TestLoader()
    
    for module_name in TEST_MODULES:
        try:
            module = __import__(module_name, fromlist=[''])
            module_tests = loader.loadTestsFromModule(module)
            
            print(f"\n{module_name}:")
            for test_group in module_tests:
                if hasattr(test_group, '_tests'):
                    for test in test_group._tests:
                        print(f"  {test.id()}")
        except Exception as e:
            print(f"\033[91mError loading {module_name}: {e}\033[0m")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run PC-E500 web emulator tests')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet output (dots only)')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List all available tests')
    parser.add_argument('test', nargs='?',
                        help='Run a specific test (e.g., tests.test_keyboard_handler.TestPCE500KeyboardHandler.test_key_press_release)')
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
        return 0
    
    if args.test:
        return run_single_test(args.test)
    
    # Determine verbosity
    verbosity = 2  # Default
    if args.quiet:
        verbosity = 1
    elif args.verbose:
        verbosity = 2
    
    return run_tests(verbosity)


if __name__ == '__main__':
    sys.exit(main())