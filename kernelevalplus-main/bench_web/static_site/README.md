# KernelEvalPlus Benchmark - Static Site

This directory contains a static HTML version of the benchmark visualization dashboard that can be deployed to GitHub Pages.

## Files

- `index.html` - Main HTML file with embedded JavaScript
- `data/` - Data directory (generated)
  - `experiments.csv` - Experimental results from CSV
  - `baseline.json` - GGML baseline performance data
- `generate_static.py` - Script to generate/update data files
- `deploy_ghpages.sh` - Deployment script for GitHub Pages

## Local Preview

To preview the site locally:

```bash
# 1. Generate data files
python3 generate_static.py

# 2. Open in browser
# Method 1: Simple Python HTTP server
cd /home/qinhaiyan/kernelevalplus/bench_web/static_site
python3 -m http.server 8000
# Then open http://localhost:8000
```

## Deploy to GitHub Pages

### Manual deployment

```bash
# 1. Generate data
python3 bench_web/static_site/generate_static.py

# 2. Create/copy to gh-pages branch
git checkout --orphan gh-pages
git rm -rf .
cp bench_web/static_site/index.html .
cp -r bench_web/static_site/data .
echo "" > .nojekyll
git add .
git commit -m "Deploy benchmark site"
git push origin gh-pages

# 3. Return to main branch
git checkout main
```

### Using GitHub Actions (recommended)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
    paths:
      - 'bench_web/static_site/**'
      - 'KERNELEVAL-exp/three_models_with_baseline_comparison.csv'
      - 'core/tools/baseline_data_compact.json'

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate static data
        run: python3 bench_web/static_site/generate_static.py

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./bench_web/static_site
```

## Data Sources

1. **Experiments CSV**: Contains LLM kernel performance results
2. **Baseline JSON**: GGML baseline performance data
   - **Laptop**: RTX 4070, RTX 5070
   - **Desktop**: RTX 4090
   - **Server**: H800, A100, A800
