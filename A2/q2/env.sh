#!/bin/bash

# 1. Load Python module (adjust 'python/3.9' to the version on your cluster)
module load python/3.9 2>/dev/null || echo "Python module load failed, using default system python"

# 2. Create a virtual environment in the current directory
echo "Creating virtual environment 'forest_env'..."
python3 -m venv forest_env

# 3. Activate the environment
source forest_env/bin/activate

# 4. Upgrade pip (always a good practice)
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install dependencies directly
echo "Installing dependencies..."
# forest_fire.py only uses standard libraries, so no installation is strictly needed right now.
# If you add external packages later, you can list them directly below like this:
# pip install numpy scipy networkx

echo "------------------------------------------------"
echo "Setup complete. To use this environment, run:"
echo "source forest_env/bin/activate"
echo "------------------------------------------------"
