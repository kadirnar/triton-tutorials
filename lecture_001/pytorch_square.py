import torch

def time_pytorch_function(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)

a = torch.tensor([1., 2., 3.])
b = torch.randn(1000,1000).cuda()

def pt_square_v0(a):
    return torch.square(a)

def pt_square_v1(a):
    return a*a

def pt_square_v2(a):
    return a**2

time_pytorch_function(pt_square_v0, b)
time_pytorch_function(pt_square_v1, b)
time_pytorch_function(pt_square_v2, b)

print("Profiling torch.square")

with torch.profiler.profile() as prof:
    pt_square_v0(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("Profiling a*a")

with torch.profiler.profile() as prof:
    pt_square_v1(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("Profiling a**2")

with torch.profiler.profile() as prof:
    pt_square_v0(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
