import numpy as np
import pytest
import argparse
import unittest.mock as mock

import os, sys
sys.path.append([".", "../", "./src/bin/", "../src/bin/"])

from src.bin.generate import main

def test_low_T():
    with mock.patch('sys.argv', ["main", "--actf", "tanh", "--n_seed", "2", "--epochs", "100", "--model_dir", "./data_assets/models", "--r", "2", "--K", "16", "--M_mu", "0", "--M_sig", "1", "--W_mu", "0", "--W_sig", "1", "--Z_mu", "0", "--Z_sig", "1", "--bs", "4", "--lr", "0.001", "--Wp_sig", "0.1", "--project", "test"]):
        parser = argparse.ArgumentParser()
        parser.add_argument('--actf', type=str, default='tanh')
        parser.add_argument('--n_seed', type=int, default=2)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--model_dir', type=str, default='./data_assets/models')
        parser.add_argument('--r', type=int, default=2)
        parser.add_argument('--K', type=int, default=16)
        parser.add_argument('--M_mu', type=float, default=0)
        parser.add_argument('--M_sig', type=float, default=1)
        parser.add_argument('--W_mu', type=float, default=0)
        parser.add_argument('--W_sig', type=float, default=1)
        parser.add_argument('--Z_mu', type=float, default=0)
        parser.add_argument('--Z_sig', type=float, default=1)
        parser.add_argument('--bs', type=int, default=4)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--Wp_sig', type=float, default=0.1)
        parser.add_argument('--project', type=str, default='test')
        args = parser.parse_args()

        main(args)
        assert os.path.exists("./data_assets/models")

def test_high_T():
    with mock.patch('sys.argv', ["main", "--actf", "tanh", "--n_seed", "2", "--epochs", "100", "--model_dir", "./data_assets/models", "--r", "2", "--K", "16", "--M_mu", "0", "--M_sig", "1", "--W_mu", "0", "--W_sig", "1", "--Z_mu", "0", "--Z_sig", "1", "--bs", "4", "--lr", "1000000", "--Wp_sig", "0.1", "--project", "test"]):
        parser = argparse.ArgumentParser()
        parser.add_argument('--actf', type=str, default='tanh')
        parser.add_argument('--n_seed', type=int, default=2)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--model_dir', type=str, default='./data_assets/models')
        parser.add_argument('--r', type=int, default=2)
        parser.add_argument('--K', type=int, default=16)
        parser.add_argument('--M_mu', type=float, default=0)
        parser.add_argument('--M_sig', type=float, default=1)
        parser.add_argument('--W_mu', type=float, default=0)
        parser.add_argument('--W_sig', type=float, default=1)
        parser.add_argument('--Z_mu', type=float, default=0)
        parser.add_argument('--Z_sig', type=float, default=1)
        parser.add_argument('--bs', type=int, default=4)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--Wp_sig', type=float, default=0.1)
        parser.add_argument('--project', type=str, default='test')
        args = parser.parse_args()

        main(args)
        assert os.path.exists("./data_assets/models")
