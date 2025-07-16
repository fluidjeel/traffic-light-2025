import os
import sys

# The path we know should be correct
site_packages_path = r"D:\algo-2025\.venv\Lib\site-packages"
fyers_api_path = os.path.join(site_packages_path, "fyers_api")

print(f"--- Checking for fyers_api at: {fyers_api_path} ---")

if os.path.isdir(fyers_api_path):
    print("SUCCESS: 'fyers_api' directory found.")
    
    # Check for the __init__.py file which makes it an importable package
    init_file_path = os.path.join(fyers_api_path, "__init__.py")
    if os.path.isfile(init_file_path):
        print(f"SUCCESS: '__init__.py' found at {init_file_path}")
        print("This directory IS a valid Python package.")
    else:
        print(f"ERROR: '__init__.py' is MISSING from the fyers_api directory.")
        print("The package is corrupted. Please reinstall with --no-cache-dir again.")
        
    print("\n--- Listing contents of fyers_api directory ---")
    try:
        contents = os.listdir(fyers_api_path)
        for item in contents:
            print(f"- {item}")
    except Exception as e:
        print(f"Could not list directory contents: {e}")

else:
    print("CRITICAL ERROR: 'fyers_api' directory NOT found at the expected location.")

print("\n--- Attempting import again ---")
try:
    from fyers_api import fyersModel
    print("\nSUCCESS: The import worked this time!")
except ModuleNotFoundError as e:
    print(f"\nFAILURE: Still getting ModuleNotFoundError: {e}")
except Exception as e:
    print(f"\nFAILURE: Got a different error during import: {e}")