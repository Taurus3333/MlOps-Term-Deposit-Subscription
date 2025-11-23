"""
Test Runner Script
Runs all tests with coverage reporting
"""
import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run pytest with coverage"""
    print("=" * 80)
    print("Running Bank Marketing ML System Tests")
    print("=" * 80)
    
    # Install test dependencies
    print("\n[1/4] Installing test dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "pytest", "pytest-cov", "coverage"
    ])
    
    # Run tests with coverage
    print("\n[2/4] Running tests with coverage...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--tb=short"
    ])
    
    # Generate coverage report
    print("\n[3/4] Generating coverage report...")
    subprocess.run([sys.executable, "-m", "coverage", "report"])
    
    # Summary
    print("\n[4/4] Test Summary")
    print("=" * 80)
    
    if result.returncode == 0:
        print("✓ All tests passed!")
        print(f"✓ Coverage report generated: {Path('htmlcov/index.html').absolute()}")
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
    
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
