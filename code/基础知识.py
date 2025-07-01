import torch

tensor1 = torch.tensor([[4, 5, 6], [1, 2, 3]], dtype=torch.float32)
tensor2 = torch.randn(2, 3)
tensor3 = torch.zeros(3, 2)
print(tensor1, tensor2, tensor3)