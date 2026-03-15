#!/bin/bash

# 1. Load Python module (HPCs usually require this)
# You can change 'python/3.9' to the version available on your cluster
module load python/3.9 2>/dev/null || echo "Python module load failed, using default system python"

# 2. Create a virtual environment in the current directory
echo "Creating virtual environment..."
python3 -m venv forest_env

# 3. Activate the environment
source forest_env/bin/activate

# 4. Upgrade pip (good practice)
pip install --upgrade pip

# 5. Install requirements (currently empty, but here for future use)
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    touch requirements.txt
    echo "No external libraries detected in forest_fire.py. Environment is ready."
fi

echo "------------------------------------------------"
echo "Setup complete. To use this environment, run:"
echo "source forest_env/bin/activate"
echo "------------------------------------------------"
