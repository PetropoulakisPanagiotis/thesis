import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([2, 128, 120, 160], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(128, 1, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
print("hi\n")
out.backward(torch.randn_like(out))
torch.cuda.synchronize()

