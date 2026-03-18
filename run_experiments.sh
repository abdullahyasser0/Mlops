#!/usr/bin/env bash
# Run all 5 MLflow experiments for Assignment 3

echo "=== Run 1: lr=0.1  batch=64  (high learning rate - unstable?) ==="
python train.py --learning_rate 0.1   --batch_size 64  --epochs 5 --run_name "lr0.1_bs64"

echo "=== Run 2: lr=0.01 batch=64  (baseline) ==="
python train.py --learning_rate 0.01  --batch_size 64  --epochs 5 --run_name "lr0.01_bs64"

echo "=== Run 3: lr=0.001 batch=64 (low learning rate - slow convergence?) ==="
python train.py --learning_rate 0.001 --batch_size 64  --epochs 5 --run_name "lr0.001_bs64"

echo "=== Run 4: lr=0.01 batch=32  (smaller batch) ==="
python train.py --learning_rate 0.01  --batch_size 32  --epochs 5 --run_name "lr0.01_bs32"

echo "=== Run 5: lr=0.01 batch=128 (larger batch) ==="
python train.py --learning_rate 0.01  --batch_size 128 --epochs 5 --run_name "lr0.01_bs128"

echo "=== All runs complete. Open http://localhost:5000 to analyze results. ==="
