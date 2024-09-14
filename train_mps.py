import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from model.net import Net

_device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(_device)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size))

def cleanup():
    dist.destroy_process_group()

def train(args, rank, model, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.nll_loss(logits, y, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(x)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

def test(model, rank, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            ddp_loss[0] += F.nll_loss(logits, y, reduction='sum').item()
            pred = logits.argmax(dim=1, keepdim=True) 
            ddp_loss[1] += pred.eq(y.view_as(pred)).sum().item()
            ddp_loss[2] += len(y)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))

def run(args, rank, world_size): 
    setup(rank, world_size)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST('data/', train=True, download=True,
                        transform=transform)
    testset = datasets.MNIST('data/', train=False,
                        transform=transform)

    sampler_train = DistributedSampler(trainset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler_test = DistributedSampler(testset, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size,
                    'sampler': sampler_train}
    test_kwargs = {'batch_size': args.test_batch_size,
                   'sampler': sampler_test}

    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    init_start_event = torch.mps.Event(enable_timing=True)
    init_end_event = torch.mps.Event(enable_timing=True)

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()

    for epoch in range(1, args.epochs + 1):
        train(args, rank, model, train_loader, optimizer, epoch, sampler=sampler_train)
        test(model, rank, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"Event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        dist.barrier()
        states = model.state_dict()
        torch.save(states, "mnist_cnn.pt")

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"mps device count: {torch.mps.device_count()}")
    RANK = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    run(args, rank=RANK, world_size=WORLD_SIZE)
