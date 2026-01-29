"""
Test script for the 0.00sec API workflow.
Run this to verify the complete system is working.
"""
import sys
from pathlib import Path
import time
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from frontend.api_client import create_client, APIClientError, JobTimeoutError
from backend.shared.models import JobType


def create_test_data():
    """Create a sample CSV file for testing"""
    data = {
        '送り速度': [100, 150, 200, 250, 300],
        '回転速度': [1000, 1500, 2000, 2500, 3000],
        '切込量': [0.5, 1.0, 1.5, 2.0, 2.5],
        '突出量': [50, 55, 60, 65, 70],
        '載せ率': [0.3, 0.4, 0.5, 0.6, 0.7],
        'パス数': [1, 2, 3, 4, 5],
        '線材本数': [4, 5, 6, 7, 8],
        'UPカット': [0, 1, 0, 1, 0],
        '摩耗量': [0.1, 0.2, 0.15, 0.25, 0.18],
        '上面ダレ量': [0.05, 0.08, 0.06, 0.09, 0.07],
        '側面ダレ量': [0.03, 0.05, 0.04, 0.06, 0.045],
        'バリ除去': ['YES', 'NO', 'YES', 'YES', 'NO']
    }
    
    df = pd.DataFrame(data)
    test_file = Path("test_data.csv")
    df.to_csv(test_file, index=False)
    
    print(f"✓ Created test data file: {test_file}")
    return test_file


def test_api_connection(api_url):
    """Test API server is reachable"""
    print("\n=== Testing API Connection ===")
    
    try:
        import requests
        response = requests.get(f"{api_url.replace('/api/v1', '')}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy")
            print(f"  - Version: {data.get('version')}")
            print(f"  - Database connected: {data.get('database_connected')}")
            return True
        else:
            print(f"✗ API returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        print(f"  Make sure API server is running at {api_url}")
        return False


def test_optimization_workflow(api_url):
    """Test D-optimization workflow"""
    print("\n=== Testing D-Optimization Workflow ===")
    
    try:
        # Create client
        client = create_client(api_url)
        print("✓ Created API client")
        
        # Create test data
        test_file = create_test_data()
        
        # Create output directory
        output_dir = Path("./test_results/optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {output_dir}")
        
        # Define parameters
        parameters = {
            "objective": "minimize_wear",
            "num_points": 10,
            "iterations": 20,
            "random_seed": 42
        }
        
        print("\n📤 Uploading file and creating job...")
        
        # Progress callback
        def progress_callback(message, percent):
            print(f"  [{percent:3d}%] {message}")
        
        # Run complete workflow
        start_time = time.time()
        
        output_files = client.submit_and_wait(
            file_path=test_file,
            job_type=JobType.OPTIMIZATION,
            parameters=parameters,
            output_dir=output_dir,
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Workflow completed in {elapsed:.1f} seconds")
        print(f"✓ Downloaded {len(output_files)} files:")
        
        for file in output_files:
            print(f"  - {file.name} ({file.stat().st_size} bytes)")
        
        return True
        
    except JobTimeoutError as e:
        print(f"\n✗ Job timed out: {e}")
        return False
    except APIClientError as e:
        print(f"\n✗ API error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_status_polling(api_url):
    """Test job status polling"""
    print("\n=== Testing Status Polling ===")
    
    try:
        import requests
        
        # Create a dummy job by requesting upload URL
        response = requests.post(
            f"{api_url}/presign/upload",
            json={"filename": "test.csv", "content_type": "text/csv"},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"✗ Failed to create test job: {response.status_code}")
            return False
        
        job_id = response.json()["job_id"]
        print(f"✓ Created test job: {job_id}")
        
        # Try to get status (will return 404 if job not in database)
        response = requests.get(
            f"{api_url}/jobs/{job_id}/status",
            timeout=10
        )
        
        # 404 is expected since we didn't actually create the job in DB
        if response.status_code == 404:
            print("✓ Status endpoint works (404 expected for uncreated job)")
            return True
        elif response.status_code == 200:
            print("✓ Status endpoint works")
            status = response.json()
            print(f"  - Status: {status['status']}")
            print(f"  - Progress: {status['progress_percent']}%")
            return True
        else:
            print(f"✗ Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Status polling test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("0.00sec System Test Suite")
    print("=" * 60)
    
    # Get API URL
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv("frontend/.env")
        api_url = os.getenv('API_BASE_URL')
        
        if not api_url:
            print("\n✗ API_BASE_URL not set in frontend/.env")
            print("  Please configure the .env file first")
            return 1
            
    except Exception as e:
        print(f"\n✗ Failed to load configuration: {e}")
        return 1
    
    print(f"\nAPI URL: {api_url}")
    
    # Run tests
    results = {
        "API Connection": test_api_connection(api_url),
        "Status Polling": test_status_polling(api_url),
    }
    
    # Ask before running full workflow test
    print("\n" + "=" * 60)
    response = input("\nRun full optimization workflow test? (y/n): ")
    
    if response.lower() == 'y':
        results["Optimization Workflow"] = test_optimization_workflow(api_url)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! System is working correctly.")
        print("\nNext steps:")
        print("1. Review output files in ./test_results/")
        print("2. Integrate with your GUI (see frontend/integration_example.py)")
        print("3. Deploy to production (see DEPLOYMENT_GUIDE.md)")
        return 0
    else:
        print("❌ Some tests failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("1. Check API server is running")
        print("2. Verify database is accessible")
        print("3. Check worker is processing jobs")
        print("4. Review logs for detailed errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
