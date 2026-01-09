#!/bin/bash

# MARL Warehouse LGV Optimization - Quick Start Script

echo "🤖 MARL Warehouse LGV Optimization System 🤖"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create necessary directories
mkdir -p models data results

# Launch Streamlit app
echo ""
echo "🚀 Launching Streamlit Dashboard..."
echo "📱 Open your browser at: http://localhost:8501"
echo ""
streamlit run app.py
