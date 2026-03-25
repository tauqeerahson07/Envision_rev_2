#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

cd EnvisionBackend
# Collect static files
python manage.py collectstatic --no-input

# Run migrations
python manage.py makemigrations
python manage.py migrate