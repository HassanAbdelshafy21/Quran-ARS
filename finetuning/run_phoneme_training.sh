#!/bin/bash
# Crash-resilient phoneme training: resume from the latest checkpoint on each (re)start,
# so a transient CUDA fault just continues instead of losing the run. Stops at MAX steps.
cd /home/pickles/Coding/Quran-ARS
source /home/pickles/miniconda3/etc/profile.d/conda.sh; conda activate mlaudio
unset HF_HUB_OFFLINE
OUT=finetuning/checkpoints_phoneme
MAX=30000
for attempt in $(seq 1 40); do
  latest=$(ls -d $OUT/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1)
  if [ -n "$latest" ] && [ "$latest" -ge "$MAX" ]; then echo "=== TRAINING COMPLETE at step $latest ==="; break; fi
  resume=""; [ -n "$latest" ] && resume="--resume_from $OUT/checkpoint-$latest"
  echo "=== attempt $attempt : resume from ${latest:-scratch} ==="
  python3 finetuning/train_phoneme_recognizer.py \
    --max_steps $MAX --batch_size 6 --grad_accum 3 --lr 3e-4 --save_steps 500 \
    --output_dir $OUT $resume
  code=$?
  echo "=== run exited code $code ==="
  [ $code -eq 0 ] && grep -q "saved final" "$OUT"/../../*/phoneme_train2.log 2>/dev/null
  sleep 15   # let the GPU/driver settle before retrying
done
echo "=== wrapper done ==="
