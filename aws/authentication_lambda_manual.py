import json
import os
import boto3
from fyers_apiv3 import fyersModel

# This script is designed to be run as an AWS Lambda function.
# It requires the 'fyers-apiv3' and 'boto3' libraries, which should be
# included in your Lambda deployment package.

# --- Environment Variables ---
# These must be set in your Lambda function's configuration in the AWS console.
# Example: CLIENT_ID = "your-fyers-client-id"
CLIENT_ID = os.environ.get('FYERS_CLIENT_ID')
SECRET_KEY = os.environ.get('FYERS_SECRET_KEY')
REDIRECT_URI = os.environ.get('FYERS_REDIRECT_URI')
# The name of the secret in AWS Secrets Manager where the token will be stored.
TOKEN_SECRET_NAME = os.environ.get('TOKEN_SECRET_NAME')


def lambda_handler(event, context):
    """
    AWS Lambda handler function.

    :param event: The event dictionary passed to the Lambda function.
                  Expected to contain: {"auth_code": "your_manual_auth_code"}
    :param context: The Lambda runtime information.
    :return: A dictionary with the status of the operation.
    """
    print("Authentication Lambda (Manual) started.")

    # 1. Retrieve the manual auth_code from the Lambda's input event
    auth_code = event.get('auth_code')
    if not auth_code:
        print("Error: 'auth_code' not found in the input event.")
        return {
            'statusCode': 400,
            'body': json.dumps({'status': 'error', 'message': "Auth code is required."})
        }

    # 2. Check if all environment variables are set
    if not all([CLIENT_ID, SECRET_KEY, REDIRECT_URI, TOKEN_SECRET_NAME]):
        print("Error: One or more required environment variables are not set.")
        return {
            'statusCode': 500,
            'body': json.dumps({'status': 'error', 'message': "Missing environment variable configuration."})
        }

    try:
        # 3. Initialize the Fyers session model
        session = fyersModel.SessionModel(
            client_id=CLIENT_ID,
            secret_key=SECRET_KEY,
            redirect_uri=REDIRECT_URI,
            response_type='code',
            grant_type='authorization_code'
        )
        session.set_token(auth_code)

        # 4. Generate the access token from the auth_code
        response = session.generate_token()

        if "access_token" not in response:
            print(f"Error generating token from Fyers API: {response.get('message')}")
            raise Exception(f"Fyers API Error: {response.get('message')}")

        access_token = response["access_token"]
        print("Successfully generated new access token from Fyers.")

        # 5. Store the new access token securely in AWS Secrets Manager
        secrets_client = boto3.client('secretsmanager')
        
        secret_string = json.dumps({'access_token': access_token})

        secrets_client.put_secret_value(
            SecretId=TOKEN_SECRET_NAME,
            SecretString=secret_string
        )
        print(f"Successfully stored the new access token in Secrets Manager: {TOKEN_SECRET_NAME}")

        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'success', 'message': 'Access token generated and stored successfully.'})
        }

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'status': 'error', 'message': str(e)})
        }

# --- Deployment Notes ---
# 1. Create a Lambda function in the AWS Console.
# 2. Package this script along with the 'fyers-apiv3' library into a zip file and upload it.
# 3. Set the following Environment Variables in the Lambda configuration:
#    - FYERS_CLIENT_ID: Your Fyers App Client ID
#    - FYERS_SECRET_KEY: Your Fyers App Secret Key
#    - FYERS_REDIRECT_URI: Your Fyers App Redirect URI
#    - TOKEN_SECRET_NAME: The name of the secret you created in AWS Secrets Manager to store the token.
# 4. IAM Role: Ensure the Lambda's execution role has permission to `secretsmanager:PutSecretValue`.
# 5. To run, manually trigger the Lambda with a test event like:
#    {
#      "auth_code": "paste_your_fresh_auth_code_here"
#    }
