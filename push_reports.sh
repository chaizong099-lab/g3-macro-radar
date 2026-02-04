#!/bin/bash
set -e

echo "🚀 Pushing reports to GitHub..."

git config --global user.email "actions@github.com"
git config --global user.name "GitHub Actions Bot"

git add reports/latest.json
git commit -m "📊 Auto update macro dashboard data" || echo "No changes to commit"

git push origin main
