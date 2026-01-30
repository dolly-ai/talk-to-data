"""
Simple test script for Talk to Data API
Run: python test_api.py
"""

import requests
import json
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

API_URL = "http://localhost:5000/api"

def print_success(message):
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_error(message):
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_info(message):
    print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")

def test_health_check():
    """Test health check endpoint"""
    print_info("Testing health check endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print_success("Health check passed")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False

def test_get_tables():
    """Test get tables endpoint"""
    print_info("Testing get tables endpoint...")
    try:
        response = requests.get(f"{API_URL}/tables")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Found tables: {', '.join(data['tables'])}")
            return True
        else:
            print_error(f"Get tables failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Get tables failed: {str(e)}")
        return False

def test_query(question):
    """Test query endpoint with a specific question"""
    print_info(f"Testing query: '{question}'")
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Query executed successfully")
            print(f"  SQL: {data['sql_query']}")
            print(f"  Results: {data['row_count']} rows")
            print(f"  Insights: {data['insights'][:100]}...")
            return True
        else:
            error_msg = response.json().get('error', 'Unknown error')
            print_error(f"Query failed: {error_msg}")
            return False
    except Exception as e:
        print_error(f"Query failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print(f"{Fore.YELLOW}Talk to Data - API Test Suite{Style.RESET_ALL}")
    print("="*60 + "\n")
    
    tests = []
    
    # Test 1: Health check
    tests.append(("Health Check", test_health_check()))
    
    # Test 2: Get tables
    tests.append(("Get Tables", test_get_tables()))
    
    # Test 3: Sample queries
    sample_questions = [
        "What are the top 5 products by revenue?",
        "Show me sales by region",
        "Which category has the highest sales?"
    ]
    
    for question in sample_questions:
        tests.append((f"Query: {question[:30]}...", test_query(question)))
    
    # Summary
    print("\n" + "="*60)
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    print(f"\nTest Results: {passed}/{total} passed")
    
    if passed == total:
        print_success("All tests passed! 🎉")
    else:
        print_error(f"{total - passed} test(s) failed")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    print(f"\n{Fore.CYAN}Make sure your backend is running on {API_URL}{Style.RESET_ALL}\n")
    input("Press Enter to start tests...")
    run_all_tests()
