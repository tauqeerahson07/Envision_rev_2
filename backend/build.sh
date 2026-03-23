#!/bin/bash

# Ensure pip, setuptools and wheel are up-to-date so pkg_resources is available
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

cd EnvisionBackend
# Apply migrations
python manage.py migrate --no-input

# Collect static files
python manage.py collectstatic --no-input