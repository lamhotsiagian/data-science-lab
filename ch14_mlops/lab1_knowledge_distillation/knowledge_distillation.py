import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

def create_teacher_model(input_dim, hidden_dims=[256, 128, 64], output_dim=10):
    layers = []
    prev_dim = input_dim
    for hd in hidden_dims:
        layers.append(nn.Linear(prev_dim, hd))
        layers.append(nn.ReLU())
        prev_dim = hd
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

def create_student_model(input_dim, hidden_dim=32, output_dim=10):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )

def distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    soft_targets_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature**2)
    label_loss = F.cross_entropy(student_logits, true_labels)
    return alpha * soft_targets_loss + (1 - alpha) * label_loss

def train_teacher(model, train_loader, epochs, lr, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            optimizer.step()
    return model

def train_student_standard(model, train_loader, epochs, lr, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            optimizer.step()
    return model

def train_student_distilled(student, teacher, train_loader, epochs, lr, temperature, alpha, device='cpu'):
    student, teacher = student.to(device), teacher.to(device)
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(X)
            student_logits = student(X)
            loss = distillation_loss(student_logits, teacher_logits, y, temperature, alpha)
            loss.backward()
            optimizer.step()
    return student

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, X, device='cpu', n_runs=10):
    model = model.to(device)
    X = X.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5): model(X)
    
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(X)
    return (time.time() - start) / n_runs

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0

def compare_distillation(train_loader, test_loader, input_dim, output_dim, epochs, device='cpu'):
    teacher = create_teacher_model(input_dim, output_dim=output_dim)
    student_std = create_student_model(input_dim, output_dim=output_dim)
    student_dist = create_student_model(input_dim, output_dim=output_dim)
    
    train_teacher(teacher, train_loader, epochs, 0.01, device)
    train_student_standard(student_std, train_loader, epochs, 0.01, device)
    train_student_distilled(student_dist, teacher, train_loader, epochs, 0.01, 3.0, 0.5, device)
    
    return {
        "teacher_acc": evaluate_model(teacher, test_loader, device),
        "student_std_acc": evaluate_model(student_std, test_loader, device),
        "student_dist_acc": evaluate_model(student_dist, test_loader, device),
        "teacher_params": count_parameters(teacher),
        "student_params": count_parameters(student_std)
    }

def plot_distillation_comparison(results):
    fig, ax = plt.subplots()
    ax.bar(['Teacher', 'Student (Standard)', 'Student (Distilled)'], 
           [results['teacher_acc'], results['student_std_acc'], results['student_dist_acc']])
    ax.set_ylabel('Accuracy')
    return fig
