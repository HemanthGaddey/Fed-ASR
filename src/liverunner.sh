#!/bin/bash

# Get the script name from command-line argument
SCRIPT="$1"

if [ -z "$SCRIPT" ]; then
  echo "Please provide the Python script name."
  exit 1
fi

# Function to get the modified time of the script file
get_modification_time() {
  stat -c %Y "$SCRIPT"
}

# Store initial modification time
last_modified=$(get_modification_time)

# Run the initial script
python3 "$SCRIPT" &

clear 

# Continuously check for changes and restart the script
while true; do
  current_modified=$(get_modification_time)
  if [ $current_modified -gt $last_modified ]; then
    clear
    echo "Restarting script..."
    pkill -f "python3 $SCRIPT"
    python3 "$SCRIPT" &
    last_modified=$current_modified
  fi
  sleep 1
done

