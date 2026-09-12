#!/bin/bash
# Mixed-residual ablation: 2 arms x 6 LRs x 2 seeds = 24 runs, 8 concurrent.
# CPU affinity is mandatory (240 cores / 8 jobs); without it rayon+OMP oversubscribe
# ~9x and no run reaches step 1.
cd /home/brayden/torchtitan
LRS="1e-4 3e-4 1e-3 3e-3 1e-2 3e-2"
JOBS=()
for seed in 0 1; do for arm in default mixed; do for lr in $LRS; do JOBS+=("$arm:$lr:$seed"); done; done; done

run_job () {
  local IFS=':'; read -r arm lr seed <<< "$1"
  local gpu="$2" lo=$((2#0)) ; lo=$((gpu*30)); local hi=$((gpu*30+29))
  CUDA_VISIBLE_DEVICES=$gpu NGPU=1 \
  OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  MODULE=torchtitan.experiments.mixed_residual CONFIG=mixed_residual_$arm \
    taskset -c ${lo}-${hi} ./run_train.sh \
    --optimizer.param-groups.0.optimizer-kwargs.lr $lr \
    --debug.seed $seed \
    --dump-folder ./outputs/mr_${arm}_${lr}_s${seed} \
    > /home/brayden/torchtitan/mr_${arm}_${lr}_s${seed}.log 2>&1
  echo "FINISHED ${arm} lr=${lr} seed=${seed} rc=$?"
}

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  pids=()
  for gpu in 0 1 2 3 4 5 6 7; do
    [ $i -ge ${#JOBS[@]} ] && break
    echo "LAUNCH ${JOBS[$i]} on gpu$gpu"
    run_job "${JOBS[$i]}" "$gpu" &
    pids+=($!)
    i=$((i+1))
  done
  for p in "${pids[@]}"; do wait $p; done
  echo "WAVE DONE ($i/${#JOBS[@]})"
done
echo "SWEEP COMPLETE"
