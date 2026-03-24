#!/usr/bin/env bash
set -o errexit  # exit on error

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🧩 Making migrations..."
python EnvisionBackend/manage.py makemigrations --noinput

echo "🗄️ Applying migrations..."
python EnvisionBackend/manage.py migrate --noinput

echo "📁 Collecting static files..."
python EnvisionBackend/manage.py collectstatic --no-input

echo "✅ Build completed successfully!"