# RL with Convolution-Augmented Attention and Gate Mechanism for Job Shop Scheduling
## Overview
This repository contains the implementation of our Deep Reinforcement Learning (DRL) framework, presented in our paper "A Deep Reinforcement Learning Framework with Convolution-Augmented Attention and Gate Mechanism for Job Shop Scheduling." Our framework leverages a novel combination of convolution-augmented attention and a gating mechanism to solve the Job Shop Scheduling Problem (JSSP), a complex combinatorial optimization problem.

## Features
Convolution-Augmented Attention: Enhances local pattern recognition within the JSSP environment.
Gate Mechanism: Facilitates effective integration of convolution and multi-head self-attention (MHSA) layers.
End-to-End Training: Utilizes a Proximal Policy Optimization (PPO) algorithm within an actor-critic framework for training.
Comprehensive Graph Representations: Captures both local and global dependencies for optimal scheduling.

## Proposed DRL Framework
The following figure depicts our advanced Deep Reinforcement Learning model, integrating a convolution-augmented attention mechanism for job shop scheduling. It details the architecture, highlighting how convolution layers and attention modules synergistically process scheduling data. 
![Conformer_gate](https://github.com/goytomdesta/Covolution_Augumented_Attention_Model-for-Job_Shop_Scheduling/assets/93028395/f9c5e43e-06a6-4416-b6dc-f65cc0a73d9a)

Moreover, the following figure presents the convolution module's architecture within our Deep Reinforcement Learning framework, detailing its structure and functional components for processing job shop scheduling data. 
![Conv_module](https://github.com/goytomdesta/Covolution_Augumented_Attention_Model-for-Job_Shop_Scheduling/assets/93028395/37a9f3f3-1f47-4636-af49-35658b3d05e8)

## Requirements
Python 3.8      
PyTorch      
Other dependencies listed in requirements.txt

## Installation
git clone https://github.com/goytomdesta     
cd Covolution_Augumented_Attention_Model-for-Job_Shop_Scheduling     
pip install -r requirements.txt

## Usage
python main.py --options     
Replace --options with specific command line arguments for training and evaluation.

## Dataset
Experiments were conducted on Taillard’s benchmark dataset. 

## Contributors
GOYTOM GEBREYESUS, GETU FELLEK, AHMED FARID, SHIGERU FUJIMURA, OSAMU YOSHIE
