import json
import os
import boto3
import time
from urllib.parse import urlparse, parse_qs
import pyotp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from fyers_apiv3 import fyersModel

# This script is designed to be run as an AWS Lambda function.
# It is highly complex to deploy due to Selenium/Chromium dependencies.
# It requires a custom Lambda Layer containing the browser, webdriver, and all Python libraries.

# --- Environment Variables ---
# These must be set in your Lambda function's configuration.
SECRET_NAME_CREDS = os.environ.get('FYERS_CREDENTIALS_SECRET_NAME') # Secret containing username, pin, totp
TOKEN_SECRET_NAME = os.environ.get('TOKEN_SECRET_NAME') # Secret where the generated token will be stored
REDIRECT_URI = os.environ.get('FYERS_REDIRECT_URI')

def get_fyers_credentials():
    """Retrieves Fyers credentials from AWS Secrets Manager."""
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=SECRET_NAME_CREDS)
    secret = json.loads(response['SecretString'])
    return secret

def lambda_handler(event, context):
    """
    AWS Lambda handler function for fully automated Fyers authentication.
    """
    print("Authentication Lambda (Automated) started.")
    
    try:
        # 1. Get credentials from Secrets Manager
        creds = get_fyers_credentials()
        CLIENT_ID = creds['client_id']
        SECRET_KEY = creds['secret_key']
        USERNAME = creds['username']
        PIN = creds['pin']
        TOTP_SECRET = creds['totp_secret']

        # 2. Generate the Fyers auth code URL
        session = fyersModel.SessionModel(
            client_id=CLIENT_ID,
            secret_key=SECRET_KEY,
            redirect_uri=REDIRECT_URI,
            response_type='code',
            grant_type='authorization_code'
        )
        auth_url = session.generate_authcode()
        print("Generated Fyers auth URL.")

        # 3. Configure and launch headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1280x1696")
        chrome_options.add_argument("--user-data-dir=/tmp/user-data")
        chrome_options.add_argument("--hide-scrollbars")
        chrome_options.add_argument("--enable-logging")
        chrome_options.add_argument("--log-level=0")
        chrome_options.add_argument("--v=99")
        chrome_options.add_argument("--single-process")
        chrome_options.add_argument("--data-path=/tmp/data-path")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--homedir=/tmp")
        chrome_options.add_argument("--disk-cache-dir=/tmp/cache-dir")
        chrome_options.binary_location = "/opt/chrome/chrome" # Path in the Lambda Layer

        driver = webdriver.Chrome(
            executable_path="/opt/chromedriver", # Path in the Lambda Layer
            chrome_options=chrome_options
        )
        print("Headless Chrome driver initialized.")

        # 4. Automate the login process
        driver.get(auth_url)
        time.sleep(2)

        # Enter Client ID
        driver.find_element(By.ID, "fy_client_id").send_keys(USERNAME)
        driver.find_element(By.ID, "clientIdSubmit").click()
        time.sleep(2)

        # Enter TOTP and PIN
        totp = pyotp.TOTP(TOTP_SECRET).now()
        driver.find_element(By.ID, "fy_totp").send_keys(totp)
        
        pin_inputs = driver.find_elements(By.XPATH, "//input[contains(@id, 'pin')]")
        for i, pin_input in enumerate(pin_inputs):
            pin_input.send_keys(PIN[i])
        
        driver.find_element(By.ID, "verifyPinSubmit").click()
        time.sleep(3) # Wait for redirect

        # 5. Capture the auth_code from the redirected URL
        redirected_url = driver.current_url
        driver.quit()
        print("Browser automation complete, driver quit.")

        parsed_url = urlparse(redirected_url)
        auth_code = parse_qs(parsed_url.query).get('auth_code', [None])[0]

        if not auth_code:
            raise Exception("Failed to capture auth_code from redirect URL.")
        print("Successfully captured auth_code.")

        # 6. Generate and store the access token (same as manual version)
        session.set_token(auth_code)
        response = session.generate_token()
        access_token = response["access_token"]
        
        secrets_client = boto3.client('secretsmanager')
        secret_string = json.dumps({'access_token': access_token})
        secrets_client.put_secret_value(SecretId=TOKEN_SECRET_NAME, SecretString=secret_string)
        
        print("Successfully generated and stored new access token.")
        return {'statusCode': 200, 'body': json.dumps({'status': 'success'})}

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({'status': 'error', 'message': str(e)})}

# --- Deployment Notes ---
# 1. This is an ADVANCED setup. It requires creating a custom AWS Lambda Layer.
# 2. The Layer must contain:
#    - A compatible version of headless Chromium browser.
#    - The matching ChromeDriver binary.
#    - All required Python libraries (selenium, pyotp, fyers-apiv3, boto3).
# 3. Create a Lambda function and attach the custom Layer.
# 4. Set the following Environment Variables:
#    - FYERS_CREDENTIALS_SECRET_NAME: The name of the secret in AWS Secrets Manager holding your Fyers login details.
#    - TOKEN_SECRET_NAME: The name of the secret where the generated token will be stored.
#    - FYERS_REDIRECT_URI: Your Fyers App Redirect URI.
# 5. IAM Role: The Lambda's role needs permissions for both `secretsmanager:GetSecretValue` and `secretsmanager:PutSecretValue`.
# 6. Increase Lambda timeout to at least 1-2 minutes to allow for browser startup and navigation.
