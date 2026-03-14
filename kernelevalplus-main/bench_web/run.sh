#!/bin/bash
# Run KernelEvalPlus Bench Web UI

cd "$(dirname "$0")/.."

# Activate conda environment (try multiple locations)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null

# Activate environment
conda activate KM-12.8 2>/dev/null || conda activate base

echo "🚀 Starting KernelEvalPlus Bench..."
echo "   URL: http://localhost:8512"
echo ""

# Install streamlit if needed
pip install streamlit plotly pandas -q 2>/dev/null

streamlit run bench_web/app.py --server.port 8512 --server.headless true
