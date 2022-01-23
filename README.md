# **Homework 1 - Programming Problem**

## **Info**

Name: The-Anh Vu-Le

Student ID: 20C13002

Course: Introduction to Deep Learning

Email: anh.vu2020@ict.jvn.edu.vn


## **Requirements**

```
torch
numpy
tqdm
tensorboard
yaml
```

## **Usage**

### Data preparation

To prepare the synthetic data for this problem, run

```
python generate_data.py
```

### **Training**

To train, run

```
python train.py --config </path/to/config> --gpus <GPU ID>
```

For example,
```
python train.py --config configs/train/A.yaml --gpus 0
```

You can remove the `gpus` flag to train on CPU.

### **Logging**

Training graph is automatically logged using Tensorboard to `runs`. To view this, run

```
tensorboard --logdir runs
```

### **Evaluation**

To evaluate a pretrained weights, run

```
python test.py --config </path/to/config> --weight </path/to/weight> --gpus <GPU ID>
```

For example,
```
python test.py --config configs/val/val.yaml --weight runs/A-2022_01_23-02_49_04/best_metric_Accuracy.pth --gpus 0
```

You can remove the `gpus` flag to evaluate on CPU.