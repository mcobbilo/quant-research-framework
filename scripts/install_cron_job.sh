#!/bin/bash

# Configuration
CRON_SCHEDULE="0 18 * * *" # Every day at 6:00 PM
PROJECT_DIR="/Users/milocobb/Desktop/Recent Swarm Papers/quant_framework"
PYTHON_BIN="$PROJECT_DIR/venv/bin/python3"
TARGET_SCRIPT="src/data/database_builder.py"
LOG_FILE="/tmp/quant_cron_market_breadth.log"

# Define the cron job line
CRON_CMD="cd '$PROJECT_DIR' && $PYTHON_BIN $TARGET_SCRIPT >> $LOG_FILE 2>&1"
NEW_CRON_JOB="$CRON_SCHEDULE $CRON_CMD"

# Check if the job already exists
(crontab -l 2>/dev/null | grep -F "$TARGET_SCRIPT") > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Cron job for $TARGET_SCRIPT already exists."
    exit 0
fi

# Append to cron natively
echo "Adding new market breadth cron job for 6:00 PM (18:00)..."
(crontab -l 2>/dev/null; echo "$NEW_CRON_JOB") | crontab -
echo "Cron job successfully installed. Log output will stream to $LOG_FILE"
