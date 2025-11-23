# Apache Airflow Orchestration - Complete Setup Guide

This guide assumes you've **NEVER** used Apache Airflow before. Every step is explained clearly.

---

## 📋 What You'll Learn

1. What Airflow is and why it matters
2. Installing Airflow on your machine
3. Setting up the Airflow database
4. Creating an admin user
5. Configuring the DAG
6. Running your ML pipeline with Airflow
7. Monitoring and debugging

---

## 🎯 Prerequisites

- Python 3.10 installed
- Your Bank Marketing project working
- Terminal/Command Prompt access
- 30-60 minutes of time

---

## Part 1: Understanding Airflow

### What is Apache Airflow?

Airflow is a **workflow orchestration tool**. Think of it as a smart scheduler that:
- Runs your ML pipeline automatically (daily, hourly, etc.)
- Shows visual status of each step
- Retries failed tasks automatically
- Sends email alerts when things break
- Keeps logs of everything

### Why Use Airflow?

**Without Airflow:**
- You manually run `python -m src.run_pipeline` every day
- If it fails at 3 AM, you don't know until morning
- No visual monitoring
- No automatic retries

**With Airflow:**
- Pipeline runs automatically at midnight
- Visual dashboard shows which tasks succeeded/failed
- Failed tasks retry automatically
- Email alerts when something breaks
- All logs in one place

### What is a DAG?

**DAG = Directed Acyclic Graph**

It's a fancy term for "workflow with steps that depend on each other."

Your ML pipeline DAG:
```
data_ingestion → data_validation → data_transformation → model_training → model_evaluation → model_pusher → success_notification
```

Each arrow means "this task must complete before the next one starts."

---

## Part 2: Installation

### Option A: Simple Installation (Recommended for Learning)

**Step 2.1: Install Airflow**

Open terminal in your project directory and run:

```bash
pip install apache-airflow==2.7.0
```

This will take 2-3 minutes. You'll see lots of packages being installed.

**Step 2.2: Verify Installation**

```bash
airflow version
```

You should see: `2.7.0`

✅ If you see this, Airflow is installed!

---

## Part 3: Initialize Airflow

### Step 3.1: Set Airflow Home Directory

**Windows (PowerShell):**
```powershell
$env:AIRFLOW_HOME = "$PWD\airflow"
```

**Windows (CMD):**
```cmd
set AIRFLOW_HOME=%CD%\airflow
```

**Mac/Linux:**
```bash
export AIRFLOW_HOME=$(pwd)/airflow
```

**What this does:** Tells Airflow to store its files in the `airflow` folder inside your project.

### Step 3.2: Initialize Airflow Database

Run:
```bash
airflow db init
```

**What happens:**
- Creates `airflow` folder
- Creates SQLite database for Airflow metadata
- Creates configuration file `airflow.cfg`

You'll see lots of output. Wait until it says "Initialization done."

**Expected output (last line):**
```
Initialization done
```

✅ Database is ready!

### Step 3.3: Create Admin User

Run this command (copy exactly):
```bash
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
```

**What this does:** Creates a user account so you can log into Airflow UI.

**Credentials:**
- Username: `admin`
- Password: `admin`

**Expected output:**
```
User "admin" created with role "Admin"
```

✅ Admin user created!

---

## Part 4: Configure Your DAG

### Step 4.1: Create DAGs Directory

```bash
mkdir -p airflow/dags
```

**What this does:** Creates a folder where Airflow looks for DAG files.

### Step 4.2: Copy DAG File

**Windows:**
```cmd
copy airflow\bank_marketing_dag.py airflow\dags\
```

**Mac/Linux:**
```bash
cp airflow/bank_marketing_dag.py airflow/dags/
```

**What this does:** Copies your ML pipeline DAG to where Airflow can find it.

### Step 4.3: Verify DAG File

Check the file exists:

**Windows:**
```cmd
dir airflow\dags
```

**Mac/Linux:**
```bash
ls airflow/dags
```

You should see: `bank_marketing_dag.py`

✅ DAG file is in place!

### Step 4.4: Test DAG for Errors

Run:
```bash
python airflow/dags/bank_marketing_dag.py
```

**Expected output:** No errors (just returns to prompt)

If you see errors, there's a syntax problem. Check the error message.

✅ DAG has no syntax errors!

---

## Part 5: Start Airflow

You need **TWO terminal windows** open.

### Terminal 1: Start Webserver

**Step 5.1:** Open first terminal in your project directory

**Step 5.2:** Set AIRFLOW_HOME again (important!)

**Windows (PowerShell):**
```powershell
$env:AIRFLOW_HOME = "$PWD\airflow"
```

**Windows (CMD):**
```cmd
set AIRFLOW_HOME=%CD%\airflow
```

**Mac/Linux:**
```bash
export AIRFLOW_HOME=$(pwd)/airflow
```

**Step 5.3:** Start webserver

```bash
airflow webserver --port 8080
```

**What you'll see:**
```
[date time] [INFO] Starting the web server on port 8080
```

**⚠️ IMPORTANT:** Keep this terminal open! Don't close it.

### Terminal 2: Start Scheduler

**Step 5.4:** Open second terminal in your project directory

**Step 5.5:** Set AIRFLOW_HOME again

**Windows (PowerShell):**
```powershell
$env:AIRFLOW_HOME = "$PWD\airflow"
```

**Windows (CMD):**
```cmd
set AIRFLOW_HOME=%CD%\airflow
```

**Mac/Linux:**
```bash
export AIRFLOW_HOME=$(pwd)/airflow
```

**Step 5.6:** Start scheduler

```bash
airflow scheduler
```

**What you'll see:**
```
[date time] [INFO] Starting the scheduler
```

**⚠️ IMPORTANT:** Keep this terminal open too!

**Wait 30 seconds** for both services to fully start.

✅ Airflow is running!

---

## Part 6: Access Airflow UI

### Step 6.1: Open Browser

Open your web browser and go to:
```
http://localhost:8080
```

### Step 6.2: Login

You'll see a login page.

Enter:
- **Username:** `admin`
- **Password:** `admin`

Click **"Sign In"**

✅ You're in Airflow UI!

### Step 6.3: Find Your DAG

You should see a list of DAGs. Look for:
```
bank_marketing_ml_pipeline
```

**If you don't see it:**
1. Wait 1-2 minutes (Airflow scans for DAGs every 30 seconds)
2. Refresh the page
3. Check that `bank_marketing_dag.py` is in `airflow/dags/` folder

✅ DAG is visible!

---

## Part 7: Run Your Pipeline

### Step 7.1: Enable the DAG

1. Find `bank_marketing_ml_pipeline` in the list
2. You'll see a toggle switch on the left (it's OFF by default)
3. Click the toggle to turn it **ON**
4. The toggle should turn blue/green

✅ DAG is enabled!

### Step 7.2: Trigger Manual Run

1. Click on the DAG name `bank_marketing_ml_pipeline`
2. You'll see the DAG graph view
3. Click the **"Play" button** (▶) in the top right
4. Select **"Trigger DAG"**
5. Click **"Trigger"** in the popup

✅ Pipeline is running!

### Step 7.3: Watch Execution

You'll see the DAG graph with colored boxes:

**Colors mean:**
- **Light Green** = Running
- **Dark Green** = Success
- **Red** = Failed
- **Gray** = Not started yet
- **Orange** = Queued

**Watch the tasks execute in order:**
1. `data_ingestion` (runs first)
2. `data_validation` (runs after ingestion)
3. `data_transformation` (runs after validation)
4. `model_training` (runs after transformation)
5. `model_evaluation` (runs after training)
6. `model_pusher` (runs after evaluation)
7. `success_notification` (runs last)

**This will take 5-15 minutes** depending on your data size.

✅ Pipeline is executing!

---

## Part 8: View Logs

### Step 8.1: Click on a Task

1. Click on any task box (e.g., `data_ingestion`)
2. A popup appears on the right

### Step 8.2: View Task Logs

1. Click **"Log"** button in the popup
2. You'll see the task's output logs
3. Scroll through to see what happened

**What you'll see:**
- Task start time
- Python output
- Any errors or warnings
- Task completion status

✅ You can see what each task did!

---

## Part 9: Understanding Task Status

### Success (All Green)

If all tasks are green:
```
✓ data_ingestion
✓ data_validation
✓ data_transformation
✓ model_training
✓ model_evaluation
✓ model_pusher
✓ success_notification
```

**Congratulations!** Your pipeline completed successfully.

### Failure (Red Task)

If a task is red:

**Step 9.1:** Click on the red task

**Step 9.2:** Click "Log" button

**Step 9.3:** Read the error message at the bottom

**Step 9.4:** Fix the issue in your code

**Step 9.5:** Clear the failed task:
1. Click on the red task
2. Click "Clear" button
3. Select "Clear"

**Step 9.6:** The task will retry automatically

---

## Part 10: Scheduled Execution

### Current Schedule

Your DAG is configured to run **daily at midnight** (`@daily`).

**What this means:**
- Every day at 00:00 (midnight), Airflow automatically triggers the pipeline
- You don't need to manually run it
- You can check the UI in the morning to see if it succeeded

### Change Schedule (Optional)

To change the schedule, edit `airflow/dags/bank_marketing_dag.py`:

Find this line:
```python
schedule_interval='@daily',
```

**Change to:**
- `@hourly` - Run every hour
- `@weekly` - Run every week
- `'0 9 * * *'` - Run at 9 AM daily
- `'*/30 * * * *'` - Run every 30 minutes
- `None` - Manual trigger only

**After changing:**
1. Save the file
2. Wait 30 seconds for Airflow to detect the change
3. Refresh the UI

---

## Part 11: Stop Airflow

When you're done:

### Step 11.1: Stop Webserver

Go to Terminal 1 (webserver) and press:
```
Ctrl + C
```

Wait for it to shut down.

### Step 11.2: Stop Scheduler

Go to Terminal 2 (scheduler) and press:
```
Ctrl + C
```

Wait for it to shut down.

✅ Airflow is stopped!

### Restart Later

To start again:
1. Open two terminals
2. Set `AIRFLOW_HOME` in both
3. Run `airflow webserver --port 8080` in Terminal 1
4. Run `airflow scheduler` in Terminal 2

---

## Part 12: Troubleshooting

### Issue: DAG Not Showing Up

**Solution 1:** Wait 1-2 minutes and refresh

**Solution 2:** Check DAG file location
```bash
# Should be here:
airflow/dags/bank_marketing_dag.py
```

**Solution 3:** Check for syntax errors
```bash
python airflow/dags/bank_marketing_dag.py
```

### Issue: Import Errors in DAG

**Error message:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Make sure you're running Airflow from your project root directory.

**Check current directory:**
```bash
pwd  # Mac/Linux
cd   # Windows
```

Should show: `.../Bank Marketing`

### Issue: Port 8080 Already in Use

**Error message:** `Address already in use`

**Solution 1:** Stop other services using port 8080

**Solution 2:** Use different port
```bash
airflow webserver --port 8081
```

Then access: `http://localhost:8081`

### Issue: Task Fails with "Data not found"

**Solution:** Run the simple orchestrator first to generate data
```bash
python -m src.run_pipeline
```

Then trigger the Airflow DAG again.

### Issue: Scheduler Not Picking Up DAG

**Solution:** Restart scheduler
1. Press Ctrl+C in scheduler terminal
2. Run `airflow scheduler` again

### Issue: Can't Login to UI

**Solution:** Recreate admin user
```bash
airflow users delete --username admin
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin
```

---

## Part 13: Advanced Features

### Email Alerts (Optional)

To get email alerts when tasks fail:

**Step 13.1:** Edit `airflow/airflow.cfg`

Find these lines:
```ini
[smtp]
smtp_host = localhost
smtp_user = airflow
smtp_password = airflow
smtp_port = 25
smtp_mail_from = airflow@example.com
```

**Step 13.2:** Update with your email settings

Example for Gmail:
```ini
[smtp]
smtp_host = smtp.gmail.com
smtp_starttls = True
smtp_ssl = False
smtp_user = your-email@gmail.com
smtp_password = your-app-password
smtp_port = 587
smtp_mail_from = your-email@gmail.com
```

**Step 13.3:** Restart Airflow

### View All DAG Runs

1. Click on DAG name
2. Click "Calendar" view
3. See all historical runs with success/failure status

### Backfill Historical Runs

Run pipeline for past dates:
```bash
airflow dags backfill bank_marketing_ml_pipeline --start-date 2024-11-01 --end-date 2024-11-23
```

---

## Part 14: Docker Alternative (Optional)

If you prefer Docker instead of local installation:

### Step 14.1: Start Airflow with Docker

```bash
cd airflow
docker-compose -f docker-compose-airflow.yml up -d
```

### Step 14.2: Wait for Initialization

Wait 2-3 minutes for services to start.

Check status:
```bash
docker-compose -f docker-compose-airflow.yml ps
```

All services should show "Up" status.

### Step 14.3: Access UI

Open browser: `http://localhost:8080`

Login:
- Username: `admin`
- Password: `admin`

### Step 14.4: Stop Docker Airflow

```bash
docker-compose -f docker-compose-airflow.yml down
```

---

## Part 15: Comparison with Simple Orchestrator

### Simple Orchestrator (`src/run_pipeline.py`)

**Pros:**
- ✅ No installation needed
- ✅ Simple to run
- ✅ Good for development

**Cons:**
- ❌ Manual execution only
- ❌ No visual monitoring
- ❌ No automatic retries
- ❌ No scheduling

**Use when:** Testing, development, quick demos

### Airflow DAG

**Pros:**
- ✅ Visual monitoring
- ✅ Automatic scheduling
- ✅ Automatic retries
- ✅ Email alerts
- ✅ Production-grade

**Cons:**
- ❌ Requires installation
- ❌ More complex setup
- ❌ Learning curve

**Use when:** Production, scheduled workflows, team collaboration

---

## 🎉 Congratulations!

You've successfully:
- ✅ Installed Apache Airflow
- ✅ Initialized the database
- ✅ Created admin user
- ✅ Configured your ML pipeline DAG
- ✅ Started Airflow webserver and scheduler
- ✅ Accessed the Airflow UI
- ✅ Triggered your pipeline
- ✅ Monitored task execution
- ✅ Viewed logs

**You now have production-grade workflow orchestration!** 🚀

---

## 📚 Quick Reference

### Start Airflow (2 Terminals)

**Terminal 1:**
```bash
export AIRFLOW_HOME=$(pwd)/airflow  # Mac/Linux
$env:AIRFLOW_HOME = "$PWD\airflow"  # Windows PowerShell
airflow webserver --port 8080
```

**Terminal 2:**
```bash
export AIRFLOW_HOME=$(pwd)/airflow  # Mac/Linux
$env:AIRFLOW_HOME = "$PWD\airflow"  # Windows PowerShell
airflow scheduler
```

### Access UI
```
http://localhost:8080
Username: admin
Password: admin
```

### Trigger DAG via CLI
```bash
airflow dags trigger bank_marketing_ml_pipeline
```

### View DAG Status
```bash
airflow dags list
```

### View Task Logs (CLI)
```bash
airflow tasks logs bank_marketing_ml_pipeline data_ingestion <execution_date>
```

---

## 📖 What to Say in Interviews

> "I implemented Apache Airflow orchestration for my ML pipeline. I created a DAG with 7 tasks that run sequentially with proper dependencies. Tasks communicate through XCom to pass artifacts between stages. The DAG is scheduled to run daily at midnight with automatic retries and email alerts. Airflow gives us visual monitoring, centralized logging, and the ability to backfill historical runs. This is how production ML teams operate—you define workflows as code, and Airflow handles scheduling, monitoring, and alerting."

---

## 🎯 Next Steps

1. **Practice:** Trigger the DAG a few times to get comfortable
2. **Explore:** Click around the UI to see different views
3. **Customize:** Change the schedule or add more tasks
4. **Deploy:** Follow CICD.md to deploy to AWS
5. **Interview:** Confidently explain your Airflow implementation

---

**You're now ready to say: "I've implemented Apache Airflow orchestration for production ML pipelines!"** 💪
