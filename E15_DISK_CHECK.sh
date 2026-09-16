#!/bin/bash
# E15 Disk Space Check Script
# Run this to verify available disk space before queuing E15

echo "=========================================="
echo "E15 DISK SPACE CHECK"
echo "=========================================="
echo ""

# Check root filesystem
echo "1. Root filesystem space:"
df -h / | tail -1
echo ""

# Check project directory
echo "2. Project directory (/project) space:"
df -h /project/ 2>/dev/null || df -h /project 2>/dev/null || echo "Cannot access /project stats"
echo ""

# Check current embeddings cache
echo "3. Current embeddings cache size:"
du -sh /project/6111759/ranaab/IGLOO/data/embeddings 2>/dev/null || echo "Embeddings directory: no data yet"
echo ""

# Check GB1 synthetic embeddings if they exist
echo "4. GB1 cached embeddings (if exist):"
find /project/6111759/ranaab/IGLOO/data/embeddings -name "*GB1*" -type f 2>/dev/null | wc -l
echo "   files found"
echo ""

# Calculate available space
echo "5. Available space in project:"
available=$(df /project 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$available" ]; then
  available_gb=$((available / 1024 / 1024))
  echo "   ${available_gb} GB available"

  # E15 Requirements
  echo ""
  echo "E15 Storage Requirements:"
  echo "   PHASE 1 (T4 lysozyme):  43 GB"
  echo "   PHASE 2 (p53):          106 GB"
  echo "   Total needed:           149 GB"
  echo ""

  if [ $available_gb -ge 160 ]; then
    echo "✅ VERDICT: PLENTY OF SPACE - Queue both phases"
  elif [ $available_gb -ge 50 ]; then
    echo "⚠️  VERDICT: LIMITED SPACE - Queue PHASE 1 only, then PHASE 2 after cleanup"
  else
    echo "❌ VERDICT: INSUFFICIENT SPACE - Use staggered approach (1 pLM at a time)"
  fi
else
  echo "   Cannot determine available space"
fi

echo ""
echo "=========================================="
