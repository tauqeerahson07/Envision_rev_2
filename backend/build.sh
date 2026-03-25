#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

cd EnvisionBackend

# Run migrations first
python manage.py makemigrations
python manage.py migrate

# Then collect static files
python manage.py collectstatic --no-input