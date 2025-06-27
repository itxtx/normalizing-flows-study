#!/usr/bin/env python3
"""
Test runner for the correctness test suite.

This script runs all correctness tests and provides a summary of results.
Any critical bugs are highlighted with the **critical-bug** tag.
"""

import subprocess
import sys
import os


def run_test_module(module_name):
    """Run a specific test module and return results."""
    print(f"\n{'='*60}")
    print(f"Running {module_name}")
    print(f"{'='*60}")
    
    cmd = ["python", "-m", "pytest", f"tests/correctness/{module_name}.py", "-v", "--tb=short"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {module_name}: {e}")
        return False


def run_all_tests():
    """Run all correctness tests."""
    print("Normalizing Flows - Algorithmic Correctness Test Suite")
    print("=" * 60)
    
    test_modules = [
        "test_invertibility",
        "test_logdet_autodiff", 
        "test_gradcheck",
        "test_distribution_preservation"
    ]
    
    results = {}
    
    for module in test_modules:
        success = run_test_module(module)
        results[module] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("CORRECTNESS TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for module, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{module:<35} {status}")
        if not success:
            all_passed = False
    
    print(f"{'='*60}")
    if all_passed:
        print("🎉 ALL CORRECTNESS TESTS PASSED")
    else:
        print("❌ SOME CORRECTNESS TESTS FAILED")
        print("   Check logs above for **critical-bug** tagged failures")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
