TFL Automated Trading System - Operations Runbook
Document Version: 1.0
System Version: v3.0 (Based on long_portfolio_simulator_v3.0.py architecture)
Infrastructure Blueprint: tfl-infrastructure.yaml

1. Introduction
This document is the official Operations Runbook for the TrafficLight-Manny (TFL) automated trading system. It serves as the single source of truth for deploying, managing, monitoring, and troubleshooting the entire AWS infrastructure that powers the trading bot.

The system is designed as a hybrid cloud architecture, leveraging a central EC2 instance for high-performance, real-time trade execution and a suite of serverless components for automation, security, and cost-effectiveness.

This runbook is intended for the system operator and assumes a basic understanding of AWS concepts.

2. Architecture Overview
The infrastructure provisioned by the tfl-infrastructure.yaml CloudFormation template follows a robust, hub-and-spoke model.

Core Compute (EC2): A single, time-sensitive EC2 instance acts as the brain, running the Python trading application during market hours. It handles the WebSocket data stream, in-memory state management, and real-time order execution.

Automation (EventBridge & Lambda): Serverless components handle all non-real-time, scheduled tasks, including starting/stopping the EC2 instance (the "lightswitch") and calculating the daily market regime.

Configuration & Security (IAM, Secrets Manager, SSM): AWS security services manage permissions and securely store sensitive credentials, removing them from the application code.

Data & State (DynamoDB): A serverless DynamoDB table acts as the transactional database, providing a durable, auditable log of every trade and state change.

Monitoring & Alerts (CloudWatch & SNS): The system's health, logs, and performance are centrally managed in CloudWatch, with critical alerts pushed via SNS.

3. Setup Instructions: Initial System Deployment
This is a one-time setup process to launch the entire infrastructure from the CloudFormation template.

3.1. Pre-Deployment Checklist
Before launching the stack, ensure you have the following prerequisites ready in the ap-south-1 (Mumbai) region:

Create an EC2 Key Pair:

Go to the EC2 service console.

Navigate to "Key Pairs" and create a new key pair (e.g., tfl-key).

Download and securely store the .pem file. You will need this to SSH into the server.

Prepare Your Application Code:

Ensure your final tfl_live_trader.py script and all its dependencies (e.g., requirements.txt, data files, symbol lists) are committed to a secure Git repository (e.g., GitHub, AWS CodeCommit). The EC2 instance will clone this repository to get the code.

3.2. Launching the CloudFormation Stack
Navigate to the AWS CloudFormation service in the console.

Click "Create stack" > "With new resources (standard)".

Under "Specify template," select "Upload a template file" and choose your tfl-infrastructure.yaml file.

Stack Name: Give the stack a clear name (e.g., TFL-Trading-System-Prod).

Parameters: Fill in the required parameters:

InstanceTypeParameter: Leave as t3.large unless you have a specific reason to change it.

MyIPAddress: Enter your current IP address in CIDR notation (e.g., 103.55.21.11/32). You can find this by searching "what is my IP" in Google. This is crucial for security.

SSHKeyName: Select the name of the EC2 Key Pair you created in the pre-deployment step.

Proceed through the next steps, acknowledge that IAM resources will be created, and click "Create stack".

The stack creation will take several minutes. You can monitor its progress in the "Events" tab.

3.3. Post-Deployment Configuration
Once the stack status shows CREATE_COMPLETE, perform these final setup steps:

Update FYERS Credentials:

Navigate to AWS Secrets Manager.

Find the secret named tfl/fyers_credentials.

Click "Retrieve secret value," then "Edit."

Replace the placeholder values with your actual Fyers client_id, secret_key, and redirect_uri.

Deploy Code to EC2:

Navigate to the EC2 console and find your newly launched "TFL-Trading-Server" instance.

Connect to it via SSH using the .pem file you downloaded.

Once connected, clone your Git repository into the /home/ec2-user directory.

git clone https://your-git-repo/tfl.git

Set up the EC2 Cron Job:

While connected via SSH, edit the crontab: crontab -e

Add the following line to the file. This will automatically start your trading application at 9:00 AM IST every weekday.

# Start the TFL trading application at 9:00 AM IST on weekdays
00 9 * * 1-5 /usr/bin/python3 /home/ec2-user/tfl/tfl_live_trader.py > /var/log/tfl_trader.log 2>&1

Subscribe to SNS Notifications:

Navigate to the SNS service console.

Find the topic named TFL-Notifications.

Go to the "Subscriptions" tab and create a new subscription. Select "Email" as the protocol and enter your email address.

You will receive a confirmation email. Click the link to confirm your subscription.

4. Operational Guide
4.1. Daily Operations (Largely Automated)
Time (IST)

System Action (Automated)

Operator Action (Manual)

8:30 AM

EventBridge starts the EC2 instance.

None.

8:45 AM

Pre-market Lambda calculates the daily regime.

Check your email/SMS for the SNS notification confirming the day's trading bias (LONGS, SHORTS, or NO_TRADES).

9:00 AM

The cron job on the EC2 starts the tfl_live_trader.py application.

None.

9:00 AM - 4:00 PM

EC2 app trades based on the regime. Logs are sent to CloudWatch.

Monitor CloudWatch Logs periodically for any ERROR or FATAL messages. Investigate immediately if any appear.

4:00 PM

Post-market Lambda generates a P&L report.

Check your email/SMS for the daily P&L report from SNS.

5:00 PM

EventBridge stops the EC2 instance to save costs.

None.

4.2. Periodic Maintenance
Weekly: Review the CloudWatch metrics for the EC2 instance (CPU Utilization, Memory Usage) to ensure the t3.large instance is appropriately sized.

Monthly (or as required): The Fyers auth_code needs to be manually generated. SSH into the EC2 instance and run the tfl_live_trader.py script manually once. It will prompt you to log in and provide the new code. The script will then run with the new credentials.

5. Cost Aspect
The system is designed to be highly cost-effective. Assuming the t3.large instance and the "lightswitch" automation:

EC2 Compute & Storage: ~₹1,500 / month

Serverless Components (Lambda, DynamoDB, etc.): Effectively free at this scale due to the generous AWS Free Tier.

Total Estimated Monthly Cost: ~₹1,550 (well within the ₹3,000 budget).

6. Failure Handling & Resiliency
This section outlines what to do when things go wrong.

Scenario

Detection Method

Immediate Action / Resolution

EC2 Instance Unresponsive

CloudWatch Alarm on Status Checks; Cannot SSH.

1. Go to the EC2 Console. <br> 2. Select the instance and choose "Instance state" > "Reboot". <br> 3. The system is resilient. When it reboots, the cron job will restart the app, which can load its state from DynamoDB.

Trading Application Crash

No new logs in CloudWatch; CloudWatch Alarm on ERROR/FATAL logs.

1. SSH into the EC2 instance. <br> 2. Check the application log file: tail -100 /var/log/tfl_trader.log <br> 3. Identify the error and debug the Python script. Restart the script manually once fixed.

Broker API/WebSocket Down

ERROR logs in CloudWatch related to fyers_api connection failures.

1. Immediately check your Fyers terminal to see if the issue is widespread. <br> 2. The Python application has basic reconnection logic. <br> 3. If the outage is prolonged, manually manage open positions via the Fyers terminal as an emergency fallback.

Pre-Market Lambda Fails

You do not receive the 8:45 AM SNS notification.

1. Go to the Lambda console and check its CloudWatch logs for errors. <br> 2. The system is designed to fail safe. If the Lambda fails, the SSM parameter remains "NO_TRADES", and the EC2 app will not trade for the day, preventing risk.

SSH Access Lost

Your IP address has changed.

1. Go to the EC2 Console > Security Groups. <br> 2. Select the TFL-SG. <br> 3. Edit the inbound rule for Port 22 and update it with your new IP address.

