# Complete CI/CD + AWS Deployment Guide - Zero to Production

This guide assumes you've **NEVER** done CI/CD or AWS deployment. Every single step is included. No skipping.

---

## 📋 What You'll Accomplish

1. Set up AWS account and IAM user
2. Install and configure AWS CLI
3. Create ECR (Docker registry) and ECS (container service)
4. Configure GitHub repository secrets
5. Set up GitHub Actions workflow
6. Deploy your ML system to AWS
7. Verify deployment and test the live API

**Time Required:** 1-2 hours (first time)

---

## 🎯 Prerequisites Checklist

- [ ] GitHub account (free)
- [ ] AWS account (free tier available)
- [ ] Project code pushed to GitHub repository
- [ ] Docker installed locally
- [ ] Terminal/Command Prompt access

---

## Part 1: AWS Account Setup

### Step 1.1: Create AWS Account

1. Open browser and go to: https://aws.amazon.com
2. Click **"Create an AWS Account"** (orange button, top right)
3. Enter email address and choose account name
4. Click **"Verify email address"**
5. Check your email and enter the verification code
6. Create a strong password
7. Choose **"Personal"** account type
8. Enter your contact information
9. Enter payment information (credit card required, but we'll use free tier)
10. Verify your phone number (SMS or voice call)
11. Choose **"Basic Support - Free"** plan
12. Click **"Complete sign up"**

**Result:** You now have an AWS account. You'll receive a confirmation email.

### Step 1.2: Sign In to AWS Console

1. Go to: https://console.aws.amazon.com
2. Click **"Root user"**
3. Enter your email address
4. Enter your password
5. Click **"Sign in"**

**Result:** You're now in the AWS Management Console.

### Step 1.3: Get Your AWS Account ID

1. In the AWS Console, click your account name (top right corner)
2. You'll see your Account ID (12 digits)
3. **Copy and save this number** - you'll need it multiple times

Example: `123456789012`

### Step 1.4: Create IAM User for GitHub Actions

**Why?** Never use your root account for deployments. Create a separate user with limited permissions.

1. In AWS Console search bar (top), type **"IAM"**
2. Click **"IAM"** from results
3. In left sidebar, click **"Users"**
4. Click **"Create user"** button (orange, top right)
5. Enter username: `github-actions-deployer`
6. Click **"Next"**
7. Select **"Attach policies directly"**
8. In the search box, type and select these policies (check the box for each):
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonECS_FullAccess`
   - `IAMReadOnlyAccess`
9. Click **"Next"**
10. Review and click **"Create user"**

**Result:** User `github-actions-deployer` is created.

### Step 1.5: Create Access Keys for IAM User

1. In the Users list, click on **"github-actions-deployer"**
2. Click **"Security credentials"** tab
3. Scroll down to **"Access keys"** section
4. Click **"Create access key"**
5. Select **"Application running outside AWS"**
6. Check the confirmation box at the bottom
7. Click **"Next"**
8. (Optional) Add description: `GitHub Actions CI/CD`
9. Click **"Create access key"**
10. **CRITICAL:** You'll see two values:
    - **Access key ID** (starts with AKIA...)
    - **Secret access key** (long random string)
11. Click **"Download .csv file"** to save them
12. **ALSO copy both values to a text file** - you'll need them soon

AKIA2AUOO3L7N5VVLIY7
/Vd5FH8+JZgp/7hE139eTTHioI3nJLcfdnlh8ROi

**⚠️ WARNING:** You can NEVER see the secret access key again after closing this page!

**Result:** You have AWS credentials for GitHub Actions.

---

## Part 2: Install and Configure AWS CLI

### Step 2.1: Install AWS CLI

**Windows:**
1. Download installer: https://awscli.amazonaws.com/AWSCLIV2.msi
2. Run the downloaded file
3. Follow installation wizard (click Next, Next, Install)
4. Close and reopen your terminal/PowerShell

**Mac:**
```bash
brew install awscli
```

**Linux (Ubuntu/Debian):**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### Step 2.2: Verify AWS CLI Installation

Open terminal and run:
```bash
aws --version
```

You should see something like: `aws-cli/2.x.x Python/3.x.x ...`

### Step 2.3: Configure AWS CLI

Run:
```bash
aws configure
```

You'll be prompted for 4 values. Enter them one by one:

```
AWS Access Key ID [None]: PASTE_YOUR_ACCESS_KEY_ID_HERE
AWS Secret Access Key [None]: PASTE_YOUR_SECRET_ACCESS_KEY_HERE
Default region name [None]: us-east-1
Default output format [None]: json
```

Press Enter after each value.

### Step 2.4: Test AWS CLI Configuration

Run:
```bash
aws sts get-caller-identity
```

You should see output like:
```json
{
    "UserId": "AIDAXXXXXXXXXXXXXXXXX",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/github-actions-deployer"
}
```

**✅ If you see this, AWS CLI is configured correctly!**

---

## Part 3: Create AWS Resources

### Step 3.1: Create ECR Repository (Docker Registry)

**What is ECR?** Amazon Elastic Container Registry - stores your Docker images in AWS.

Run this command:
```bash
aws ecr create-repository --repository-name bank-marketing-api --region us-east-1 --image-scanning-configuration scanOnPush=true
```

**Output:** You'll see JSON with a `repositoryUri`. It looks like:
```
123456789012.dkr.ecr.us-east-1.amazonaws.com/bank-marketing-api
688567278334.dkr.ecr.us-east-1.amazonaws.com/bank-marketing-api
```

**📝 SAVE THIS URI!** Copy it to your text file. You'll need it for GitHub secrets.

### Step 3.2: Get Your Default VPC ID

Run:
```bash
aws ec2 describe-vpcs --query "Vpcs[?IsDefault==\`true\`].VpcId" --output text
```

**Output:** Something like `vpc-0a1b2c3d4e5f6g7h8`
vpc-069c40c786a9f1caf

**📝 SAVE THIS VPC ID!**

### Step 3.3: Get Your Default Subnet IDs

Run:
```bash
aws ec2 describe-subnets --filters "Name=vpc-id,Values=YOUR_VPC_ID" --query "Subnets[*].SubnetId" --output text
```

Replace `YOUR_VPC_ID` with the VPC ID from previous step.

**Output:** Something like `subnet-abc123 subnet-def456`

**📝 SAVE AT LEAST TWO SUBNET IDs!** (ECS requires at least 2 subnets)
subnet-065a8e8bfb48b938c        subnet-039931f48dae447b7        subnet-0d7c63e6cf74f0dec        subnet-0fd9bec9d047bf956        subnet-002c6e2cfd8e8acb9        subnet-0e4aa91b349d8af34

### Step 3.4: Create Security Group

Run:
```bash
aws ec2 create-security-group --group-name bank-marketing-sg --description "Security group for Bank Marketing API" --vpc-id YOUR_VPC_ID
```

Replace `YOUR_VPC_ID` with your VPC ID.

**Output:** You'll get a `GroupId` like `sg-0a1b2c3d4e5f6g7h8`

**📝 SAVE THIS SECURITY GROUP ID!**

{
    "GroupId": "sg-0174a31019b51f977",
    "SecurityGroupArn": "arn:aws:ec2:us-east-1:688567278334:security-group/sg-0174a31019b51f977"
}


### Step 3.5: Allow Inbound Traffic on Port 8000

Run:
```bash
aws ec2 authorize-security-group-ingress --group-id YOUR_SECURITY_GROUP_ID --protocol tcp --port 8000 --cidr 0.0.0.0/0
```

Replace `YOUR_SECURITY_GROUP_ID` with the security group ID from previous step.

**Result:** Port 8000 is now open for your API.

### Step 3.6: Create ECS Cluster

Run:
```bash
aws ecs create-cluster --cluster-name bank-marketing-cluster --region us-east-1
```

**Result:** ECS cluster created. You'll see confirmation JSON.

### Step 3.7: Create CloudWatch Log Group

Run:
```bash
aws logs create-log-group --log-group-name /ecs/bank-marketing --region us-east-1
```

**Result:** Log group created for container logs.

### Step 3.8: Create ECS Task Execution Role

This role allows ECS to pull images from ECR and write logs.

**Step 3.8.1:** Create trust policy file

**📍 WHERE TO CREATE:** On your **local machine**, in your project root directory

Create a file named `trust-policy.json` (in your project folder):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**Step 3.8.2:** Create the role in AWS

**📍 WHERE TO RUN:** On your **local machine** (command creates role in AWS Cloud)

Run:
```bash
aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document file://trust-policy.json
```

**Step 3.8.3:** Attach AWS managed policy

**📍 WHERE TO RUN:** On your **local machine** (command updates role in AWS Cloud)

Run:
```bash
aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

**Result:** ECS task execution role is ready.

### Step 3.9: Create ECS Task Definition

**📍 WHERE TO CREATE:** On your **local machine**, in your project root directory

Create a file named `task-definition.json` (in your project folder):

```json
{
  "family": "bank-marketing-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "bank-marketing-api",
      "image": "YOUR_ECR_URI:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        {
          "name": "MLFLOW_TRACKING_URI",
          "value": "file:./mlruns"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/bank-marketing",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "api"
        }
      }
    }
  ]
}
```

**Replace these values:**
- `YOUR_ACCOUNT_ID` → Your 12-digit AWS account ID
- `YOUR_ECR_URI` → Your ECR repository URI from Step 3.1

**Register the task definition:**

**📍 WHERE TO RUN:** On your **local machine** (command registers task in AWS ECS)

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

**Result:** Task definition registered in AWS ECS.

### Step 3.10: Create ECS Service

**📍 WHERE TO RUN:** On your **local machine** (command creates service in AWS ECS)

Run this command (replace the placeholders):
```bash
aws ecs create-service \
  --cluster bank-marketing-cluster \
  --service-name bank-marketing-service \
  --task-definition bank-marketing-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[SUBNET_ID_1,SUBNET_ID_2],securityGroups=[YOUR_SECURITY_GROUP_ID],assignPublicIp=ENABLED}"
```

**Replace:**
- `SUBNET_ID_1` and `SUBNET_ID_2` → Your subnet IDs from Step 3.3
- `YOUR_SECURITY_GROUP_ID` → Your security group ID from Step 3.4

**Result:** ECS service created in AWS. It will try to run your container (will fail initially because no Docker image exists in ECR yet - that's OK, we'll push the image later via GitHub Actions).

---

**📝 IMPORTANT CLARIFICATION - Where Things Are:**

| What | Where It Lives | How You Create It |
|------|----------------|-------------------|
| `trust-policy.json` file | Your local machine (project folder) | You create it manually with text editor |
| `task-definition.json` file | Your local machine (project folder) | You create it manually with text editor |
| IAM Role | AWS Cloud | AWS CLI command from your local machine |
| Task Definition | AWS ECS (Cloud) | AWS CLI command from your local machine |
| ECS Service | AWS ECS (Cloud) | AWS CLI command from your local machine |
| Docker Image | AWS ECR (Cloud) | GitHub Actions will push it later |

**Think of it this way:**
- You create **configuration files** (JSON) on your **local computer**
- You run **AWS CLI commands** from your **local computer**
- Those commands **create resources** in **AWS Cloud**
- Your local files are just templates - the actual resources live in AWS

---

---

## Part 4: Configure GitHub Repository

### Step 4.1: Push Your Code to GitHub

If you haven't already:

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Bank Marketing ML System"

# Create repository on GitHub (via web interface)
# Then add remote and push:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 4.2: Add GitHub Secrets

**Why?** Secrets store sensitive information (AWS credentials) securely.

1. Go to your GitHub repository in browser
2. Click **"Settings"** tab (top right)
3. In left sidebar, click **"Secrets and variables"** → **"Actions"**
4. Click **"New repository secret"** button

**Add these secrets one by one:**

**Secret 1: AWS_ACCESS_KEY_ID**
- Name: `AWS_ACCESS_KEY_ID`
- Value: Paste your Access Key ID (from Part 1, Step 1.5)
- Click **"Add secret"**

**Secret 2: AWS_SECRET_ACCESS_KEY**
- Click **"New repository secret"** again
- Name: `AWS_SECRET_ACCESS_KEY`
- Value: Paste your Secret Access Key (from Part 1, Step 1.5)
- Click **"Add secret"**

**Secret 3: ECR_REPOSITORY (Optional but recommended)**
- Click **"New repository secret"** again
- Name: `ECR_REPOSITORY`
- Value: `bank-marketing-api`
- Click **"Add secret"**

**Secret 4: SONAR_TOKEN (Optional - for code quality)**
- If you want SonarQube scanning:
  1. Go to https://sonarcloud.io
  2. Sign up with GitHub
  3. Create new project
  4. Generate token
  5. Add as secret with name `SONAR_TOKEN`

**Secret 5: SONAR_HOST_URL (Optional)**
- Name: `SONAR_HOST_URL`
- Value: `https://sonarcloud.io`
- Click **"Add secret"**

**Result:** Your secrets are configured. You should see them listed (values are hidden).

---

## Part 5: Verify GitHub Actions Workflow

### Step 5.1: Check Workflow File Exists

In your repository, verify this file exists:
`.github/workflows/ci-cd.yml`

If it doesn't exist, the workflow won't run.

### Step 5.2: Update Workflow (if needed)

Open `.github/workflows/ci-cd.yml` and verify these environment variables match your setup:

```yaml
env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: bank-marketing-api
  ECS_CLUSTER: bank-marketing-cluster
  ECS_SERVICE: bank-marketing-service
  ECS_TASK_DEFINITION: bank-marketing-task
  CONTAINER_NAME: bank-marketing-api
```

If any names are different, update them.

### Step 5.3: Commit and Push Changes

If you made any changes:
```bash
git add .
git commit -m "Update CI/CD configuration"
git push origin main
```

---

## Part 6: Trigger First Deployment

### Step 6.1: Trigger GitHub Actions

**Option 1: Push to main branch**
```bash
# Make a small change (e.g., update README)
echo "# Deployment test" >> README.md
git add README.md
git commit -m "Trigger CI/CD pipeline"
git push origin main
```

**Option 2: Manual trigger (if workflow supports it)**
1. Go to your GitHub repository
2. Click **"Actions"** tab
3. Select your workflow
4. Click **"Run workflow"** button

### Step 6.2: Monitor Pipeline Execution

1. Go to **"Actions"** tab in your GitHub repository
2. Click on the running workflow
3. Watch each job execute:
   - ✅ Quality Check (tests, linting)
   - ✅ Build & Push (Docker image to ECR)
   - ✅ Deploy (Update ECS service)

**This will take 5-15 minutes for first run.**

### Step 6.3: Check for Errors

If any job fails:
1. Click on the failed job
2. Expand the failed step
3. Read the error message
4. Fix the issue in your code
5. Commit and push again

**Common issues:**
- Missing secrets → Add them in GitHub Settings
- AWS permissions → Check IAM policies
- Docker build errors → Test locally first
- ECS deployment timeout → Check task definition

---

## Part 7: Verify Deployment

### Step 7.1: Get Public IP of Your Container

Run:
```bash
aws ecs describe-tasks \
  --cluster bank-marketing-cluster \
  --tasks $(aws ecs list-tasks --cluster bank-marketing-cluster --service-name bank-marketing-service --query 'taskArns[0]' --output text) \
  --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
  --output text
```

This gives you the network interface ID. Now get the public IP:

```bash
aws ec2 describe-network-interfaces \
  --network-interface-ids YOUR_NETWORK_INTERFACE_ID \
  --query 'NetworkInterfaces[0].Association.PublicIp' \
  --output text
```

**📝 SAVE THIS PUBLIC IP!**

### Step 7.2: Test Health Endpoint

Open browser or use curl:
```bash
curl http://YOUR_PUBLIC_IP:8000/health
```

You should see:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-23T..."
}
```

**✅ If you see this, your API is live!**

### Step 7.3: Test Web UI

Open browser and go to:
```
http://YOUR_PUBLIC_IP:8000
```

You should see the Bank Marketing prediction UI.

### Step 7.4: Test Prediction Endpoint

```bash
curl -X POST http://YOUR_PUBLIC_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 300,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }'
```

You should get a prediction response.

---

## Part 8: View Logs and Monitor

### Step 8.1: View Container Logs

**Option 1: AWS Console**
1. Go to AWS Console → CloudWatch
2. Click "Log groups"
3. Click `/ecs/bank-marketing`
4. Click on the latest log stream
5. View your application logs

**Option 2: AWS CLI**
```bash
aws logs tail /ecs/bank-marketing --follow
```

### Step 8.2: Monitor ECS Service

**Check service status:**
```bash
aws ecs describe-services \
  --cluster bank-marketing-cluster \
  --services bank-marketing-service
```

Look for `runningCount: 1` and `desiredCount: 1`.

---

## Part 9: Make Updates and Redeploy

### Step 9.1: Update Your Code

Make changes to your application code.

### Step 9.2: Commit and Push

```bash
git add .
git commit -m "Update: description of changes"
git push origin main
```

### Step 9.3: Watch Automatic Deployment

1. Go to GitHub Actions tab
2. Watch the pipeline run automatically
3. New Docker image is built and pushed
4. ECS service is updated with new image
5. Old container is stopped, new one starts

**Zero downtime deployment!**

---

## Part 10: Cleanup (When Done Testing)

### To Stop Spending Money:

**Delete ECS Service:**
```bash
aws ecs update-service \
  --cluster bank-marketing-cluster \
  --service bank-marketing-service \
  --desired-count 0

aws ecs delete-service \
  --cluster bank-marketing-cluster \
  --service bank-marketing-service \
  --force
```

**Delete ECS Cluster:**
```bash
aws ecs delete-cluster --cluster bank-marketing-cluster
```

**Delete ECR Repository:**
```bash
aws ecr delete-repository \
  --repository-name bank-marketing-api \
  --force
```

**Delete Security Group:**
```bash
aws ec2 delete-security-group --group-id YOUR_SECURITY_GROUP_ID
```

**Delete Log Group:**
```bash
aws logs delete-log-group --log-group-name /ecs/bank-marketing
```

---

## 🎉 Congratulations!

You've successfully:
- ✅ Set up AWS account and IAM user
- ✅ Configured AWS CLI
- ✅ Created ECR, ECS, and supporting resources
- ✅ Configured GitHub Actions secrets
- ✅ Deployed your ML system to production
- ✅ Verified the deployment works
- ✅ Set up automated CI/CD pipeline

**You now have a production-grade MLOps system running on AWS with full CI/CD automation!**

---

## 📚 Troubleshooting Guide

### Issue: GitHub Actions fails with "AccessDenied"
**Solution:** Check that AWS secrets are correctly added in GitHub Settings → Secrets

### Issue: ECS task fails to start
**Solution:** Check CloudWatch logs for error messages. Common causes:
- Docker image not found in ECR
- Insufficient memory/CPU
- Port conflicts

### Issue: Can't access API via public IP
**Solution:** 
- Verify security group allows inbound traffic on port 8000
- Check that ECS task has public IP assigned
- Verify container is running: `aws ecs list-tasks --cluster bank-marketing-cluster`

### Issue: Docker build fails in GitHub Actions
**Solution:** Test Docker build locally first:
```bash
docker build -t bank-marketing-api:test .
docker run -p 8000:8000 bank-marketing-api:test
```

### Issue: Pipeline succeeds but API returns 500 errors
**Solution:** Check application logs in CloudWatch. Likely causes:
- Missing model files
- Environment variables not set
- Python dependencies missing

---

## 📖 Additional Resources

- AWS ECS Documentation: https://docs.aws.amazon.com/ecs/
- GitHub Actions Documentation: https://docs.github.com/en/actions
- Docker Documentation: https://docs.docker.com/
- AWS CLI Reference: https://docs.aws.amazon.com/cli/

---

**You're now ready to confidently say: "I've deployed a production ML system to AWS with full CI/CD automation!"** 🚀
