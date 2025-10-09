#!/bin/bash
# Quick FID calculation with proper class matching

echo "============================================"
echo "FID Calculation ... "
echo "============================================"
echo ""

# GENERATED="/fs/ess/PAS2136/bio_diffusion/ip-adapter_runs/samples/samples/taxabind_2gpus_58k_10samples"
GENERATED="/fs/ess/PAS2136/bio_diffusion/ip-adapter_runs/baselines/sd15-csv-birds"
REAL_ALL="/fs/ess/PAS2136/bio_diffusion/data/inat/images/train_mini"
REAL_SUBSET="/fs/ess/PAS2136/bio_diffusion/data/inat/images/train_mini_subset_taxabind"

echo "Step 1: Creating matched subset of real images"
echo "--------------------------------------------"
echo ""

python /users/PAS2136/mridul/scratchpad/evaluation-scripts/fid/prepare_subset_fid.py \
  --generated "$GENERATED" \
  --real_all "$REAL_ALL" \
  --output_real "$REAL_SUBSET" \
  --symlink

if [ $? -ne 0 ]; then
    echo "❌ Failed to create subset. Check errors above."
    exit 1
fi

echo ""
echo "Step 2: Calculating FID score"
echo "--------------------------------------------"
echo ""

python /users/PAS2136/mridul/scratchpad/evaluation-scripts/fid/calculate_fid.py \
  --real "$REAL_SUBSET" \
  --generated "$GENERATED" \
  --batch-size 128 \
  --num-workers 12

FID_EXIT_CODE=$?

# echo ""
# echo "Step 3: Cleaning up symlink directory"
# echo "--------------------------------------------"

# if [ -d "$REAL_SUBSET" ]; then
#     echo "Removing symlink directory: $REAL_SUBSET"
#     rm -rf "$REAL_SUBSET"
#     echo "✅ Cleanup complete"
# else
#     echo "⚠️  Directory not found (may have been cleaned already)"
# fi

echo ""
if [ $FID_EXIT_CODE -eq 0 ]; then
    echo "✅ Done!"
else
    echo "❌ FID calculation failed (exit code: $FID_EXIT_CODE)"
fi