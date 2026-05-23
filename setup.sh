#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Done! Run: python server.py"