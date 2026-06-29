uv run python evaluate.py output/ --ref-dir datasets/DIV2K_HR --lpips --ray-mode --ray-auto-scan

# Ablation: no merge
for n in 256 512 1024; do
  CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/DIV2K_HR --target-modulo 4 --target-remainder 0 --max-targets 200     --output-root ./output/precise_unclosed_no_merge_${n} -- --mode unclosed --num-curves ${n} --renderer-backend cubic --curve-schedule adaptive --gini-step-every 500 --gini-warmup-iter 1000 --gini-prune-max-ratio 0.20 --merge-step-every 1000 --merge-warmup-iter 50000 --merge-jaccard-min 0.35 --merge-max-pairs 24 --iterations 11999 --lr 0.1 --bezier-degree=3 --cubic-distance-samples-train 24 --cubic-distance-samples-eval 24 --no-restore-best-at-end
done

for n in 256 512 1024; do
  CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/DIV2K_HR --target-modulo 4 --target-remainder 0 --max-targets 200     --output-root ./output/precise_closed_no_merge_${n} -- --mode closed --num-curves ${n} --renderer-backend cubic --curve-schedule adaptive --gini-step-every 500 --gini-warmup-iter 1000 --gini-prune-max-ratio 0.20 --merge-step-every 1000 --merge-warmup-iter 50000 --merge-jaccard-min 0.35 --merge-max-pairs 24 --iterations 9999 --lr 0.1 --cubic-distance-samples-train 24 --cubic-distance-samples-eval 24 --no-restore-best-at-end
done

# Ablation: totally schedule free
for n in 256 512 1024; do
  CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/DIV2K_HR --target-modulo 4 --target-remainder 0 --max-targets 200     --output-root ./output/precise_unclosed_no_schedule_${n} -- --mode unclosed --num-curves ${n} --renderer-backend cubic --curve-schedule none --iterations 11999 --lr 0.1 --bezier-degree=3 --cubic-distance-samples-train 24 --cubic-distance-samples-eval 24 --no-restore-best-at-end
done

for n in 256 512 1024; do
  CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/DIV2K_HR --target-modulo 4 --target-remainder 0 --max-targets 200     --output-root ./output/precise_closed_no_schedule_${n} -- --mode closed --num-curves ${n} --renderer-backend cubic --curve-schedule none --iterations 9999 --lr 0.1 --cubic-distance-samples-train 24 --cubic-distance-samples-eval 24 --no-restore-best-at-end
done

# Ablation: add back regularization for the control points, to check if the performance drop 
for n in 256 512 1024; do
  CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/DIV2K_HR --target-modulo 4 --target-remainder 0 --max-targets 200     --output-root ./output/precise_unclosed_with_reg_${n} -- --mode unclosed --num-curves ${n} --renderer-backend cubic --curve-schedule adaptive --gini-step-every 500 --gini-warmup-iter 1000 --gini-prune-max-ratio 0.20 --merge-step-every 1000 --merge-warmup-iter 50000 --iterations 11999 --lr 0.1 --bezier-degree=3 --cubic-distance-samples-train 24 --cubic-distance-samples-eval 24 --use-reg-loss --no-restore-best-at-end
done

for n in 256 512 1024; do
  CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/DIV2K_HR --target-modulo 4 --target-remainder 0 --max-targets 200     --output-root ./output/precise_closed_with_reg_${n} -- --mode closed --num-curves ${n} --renderer-backend cubic --curve-schedule adaptive --gini-step-every 500 --gini-warmup-iter 1000 --gini-prune-max-ratio 0.20 --merge-step-every 1000 --merge-warmup-iter 50000 --iterations 9999 --lr 0.1 --cubic-distance-samples-train 24 --cubic-distance-samples-eval 24 --use-reg-loss --no-restore-best-at-end
done

# Even more ablation: check if de Casteljau dynamic sampling can help with the training, by comparing with fixed uniform sampling for training and evaluation
for n in 256 512 1024; do
  CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/DIV2K_HR --target-modulo 4 --target-remainder 0 --max-targets 200     --output-root ./output/unclosed_decasteljau_${n} -- --mode unclosed --num-curves ${n} --renderer-backend cubic --curve-schedule adaptive --gini-step-every 500 --gini-warmup-iter 1000 --gini-prune-max-ratio 0.20 --merge-warmup-iter 50000 --iterations 11999 --lr 0.1 --bezier-degree=3 --no-restore-best-at-end --cubic-flatten-method de_casteljau
  
  CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/DIV2K_HR --target-modulo 4 --target-remainder 0 --max-targets 200     --output-root ./output/closed_decasteljau_${n} -- --mode closed --num-curves ${n} --renderer-backend cubic --curve-schedule adaptive --gini-step-every 500 --gini-warmup-iter 1000 --gini-prune-max-ratio 0.20 --merge-warmup-iter 50000 --iterations 9999 --lr 0.1 --no-restore-best-at-end --cubic-flatten-method de_casteljau
done

# Now we prove that merge startegy should be default off
for s in 8 12 24 32; do
  for n in 256 512 1024; do
    CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/DIV2K_HR --target-modulo 4 --target-remainder 0 --max-targets 200     --output-root ./output/unclosed_samples_${s}_${n} -- --mode unclosed --num-curves ${n} --renderer-backend cubic --curve-schedule adaptive --gini-step-every 500 --gini-warmup-iter 1000 --gini-prune-max-ratio 0.20 --merge-warmup-iter 50000 --iterations 11999 --lr 0.1 --bezier-degree=3 --cubic-distance-samples-train ${s} --cubic-distance-samples-eval 24 --no-restore-best-at-end

    CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/DIV2K_HR --target-modulo 4 --target-remainder 0 --max-targets 200     --output-root ./output/closed_samples_${s}_${n} -- --mode closed --num-curves ${n} --renderer-backend cubic --curve-schedule adaptive --gini-step-every 500 --gini-warmup-iter 1000 --gini-prune-max-ratio 0.20 --merge-warmup-iter 50000 --iterations 9999 --lr 0.1 --cubic-distance-samples-train ${s} --cubic-distance-samples-eval 24 --no-restore-best-at-end
  done
done

uv run python evaluate.py output_kodak/ --ref-dir datasets/kodak --lpips --ray-mode --ray-auto-scan

for n in 256 512 1024; do
  CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/kodak  --max-targets 24     --output-root ./output_kodak/kodak_unclosed_${n} -- --mode unclosed --num-curves ${n} --renderer-backend cubic --curve-schedule adaptive --gini-step-every 500 --gini-warmup-iter 1000 --gini-prune-max-ratio 0.20 --merge-warmup-iter 50000 --iterations 11999 --lr 0.1 --bezier-degree=3 --no-restore-best-at-end --cubic-distance-samples-train 24 --cubic-distance-samples-eval 24

  CUDA_HOME=/usr/local/cuda-12.8 uv run python tools/ray_train_scheduler.py --workers-per-gpu 2  --target-dir datasets/kodak  --max-targets 24     --output-root ./output_kodak/kodak_closed_${n} -- --mode closed --num-curves ${n} --renderer-backend cubic --curve-schedule adaptive --gini-step-every 500 --gini-warmup-iter 1000 --gini-prune-max-ratio 0.20 --merge-warmup-iter 50000 --iterations 9999 --lr 0.1 --cubic-distance-samples-train 12 --cubic-distance-samples-eval 24 --no-restore-best-at-end
done

for id in 1 2 3 4; do
    name=$(printf "000%d" $id)
    offset=$(printf "%04d" $((id * 4)))
    CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=6 PYTORCH_ALLOC_CONF=expandable_segments:True uv run python tools/scale_sampling_psnr_study.py   --checkpoint output/precise_unclosed_no_merge_1024/${name}_${offset}/${offset}_model.pt   --target datasets/DIV2K_HR/${offset}.png   --scales 1,2,4,8,12   --sample-counts 8,12,24,32,48   --eval-iters 10   --output-dir output/psnr_scale/unclosed

    CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=6 PYTORCH_ALLOC_CONF=expandable_segments:True uv run python tools/scale_sampling_psnr_study.py   --checkpoint output/precise_closed_no_merge_1024/${name}_${offset}/${offset}_model.pt   --target datasets/DIV2K_HR/${offset}.png   --scales 1,2,4,8,12   --sample-counts 8,12,24,32,48   --eval-iters 10   --output-dir output/psnr_scale/closed
done

# Some ideal results to show the performance on special images, which are not from DIV2K or Kodak
uv run python train.py   --target datasets/special/120.image.png --mode closed --num-curves 2048 --renderer-backend cubic --curve-schedule adaptive   --gini-step-every 500   --gini-warmup-iter 1000   --gini-prune-max-ratio 0.20 --merge-warmup-iter 50000 --output-dir output/special --iterations 14999 --lr 0.1 --cubic-distance-samples-eval 24 --cubic-distance-samples-train 24

CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 uv run python train.py   --target datasets/DIV2K_HR/0016.png --mode unclosed --num-curves 2048 --renderer-backend cubic --curve-schedule adaptive   --gini-step-every 500   --gini-warmup-iter 1000   --gini-prune-max-ratio 0.20 --merge-warmup-iter 50000 --output-dir output/special --iterations 14999 --lr 0.1 --cubic-distance-samples-eval 24 --cubic-distance-samples-train 24 --bezier-degree 3

CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 uv run python train.py   --target datasets/DIV2K_HR/0004.png --mode closed --num-curves 1024 --renderer-backend cubic --curve-schedule adaptive   --gini-step-every 500   --gini-warmup-iter 1000   --gini-prune-max-ratio 0.20 --merge-warmup-iter 50000 --output-dir output/special --iterations 9999 --lr 0.1 --cubic-distance-samples-eval 24 --cubic-distance-samples-train 24