"""
Run all problems - generates all required outputs
Just run: python run_all.py
"""
import os
import torch

print('='*60)
print('Deep Learning Homework 1')
print('='*60)

# Prefer GPU if available (each problem script also does its own device selection)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device in run_all: {device}')

os.makedirs('results', exist_ok=True)

print('\n>>> Running Problem 1 (MLPs)...')
exec(open(os.path.join('assg_1', 'problem1_mlp.py')).read())

print('\n>>> Running Problem 2 (Optimization)...')
exec(open(os.path.join('assg_1', 'problem2_optimization.py')).read())

print('\n>>> Running Problem 3 (CNNs)...')
exec(open(os.path.join('assg_1', 'problem3_cnn.py')).read())

print('\n>>> Running Bonus (Transfer Learning)...')
exec(open(os.path.join('assg_1', 'bonus_transfer_learning.py')).read())

print('\n' + '='*60)
print('ALL DONE!')
print('Check results/ folder for plots and figures')
print('='*60)


