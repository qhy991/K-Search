#!/bin/bash
# Deploy to GitHub Pages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STATIC_DIR="$SCRIPT_DIR"

echo "================================================"
echo "KernelEvalPlus - GitHub Pages Deployment"
echo "================================================"

# Generate data first
echo "Generating data files..."
python3 "$STATIC_DIR/generate_static.py"

# Check if gh-pages branch exists
if git rev-parse --verify gh-pages >/dev/null 2>&1; then
    echo "Using existing gh-pages branch"
else
    echo "Creating new gh-pages branch"
    git checkout --orphan gh-pages
    git rm -rf .
    touch .nojekyll
    echo "# KernelEvalPlus Benchmark" > README.md
    git add .nojekyll README.md
    git commit -m "Initial gh-pages branch"
    git checkout main
fi

# Create temp directory for build
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

echo "Building site in $BUILD_DIR..."

# Copy static files
cp "$STATIC_DIR/index.html" "$BUILD_DIR/"
cp -r "$STATIC_DIR/data" "$BUILD_DIR/"
cp "$STATIC_DIR/.nojekyll" "$BUILD_DIR/" 2>/dev/null || echo "" > "$BUILD_DIR/.nojekyll"

# Switch to gh-pages and update
echo "Updating gh-pages branch..."
git checkout gh-pages

# Remove old files (except .git and .nojekyll)
find . -maxdepth 1 ! -name '.' ! -name '.git' ! -name '.nojekyll' -exec rm -rf {} +

# Copy new files
cp -r "$BUILD_DIR"/* .

# Add and commit
git add .
git commit -m "Update benchmark site - $(date '+%Y-%m-%d %H:%M:%S')" || echo "No changes to commit"

# Push
echo "Pushing to origin..."
git push origin gh-pages

# Return to main branch
git checkout main

echo ""
echo "================================================"
echo "Deployment complete!"
echo "================================================"
echo "Your site should be available at:"
echo "https://<username>.github.io/<repository>/"
echo ""
