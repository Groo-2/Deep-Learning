import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#
t1 = torch.linspace(1, 12, 12)
print(t1.shape) # output : 12
print(t1)

#
t2 = t1.view(3,4)
print(t2)

#
t3 = t2.clone() # deep copy
t2[1,2] = 70
print(t3)
print(t2)

# 0 : 행, 1 : 열, 2 : DEPTH
t4 = t2.unsqueeze(dim=0)
print(t4.shape) # [1, 3, 4]
print(t4)

# unsqueeze : 차원 늘리기
t5 = t3.unsqueeze(dim=0)
print(t5.shape) # [1, 3, 4]
print(t5)

# squeeze : 차원 되돌리기
t5 = t5.squeeze()
print(t5)

#
t5 = t5.unsqueeze(dim=1)
print(t5.shape) # output : [3, 1, 4]
print(t5)
t5 = t5.squeeze()
print(t5)

t5 = t5.unsqueeze(dim=2)
print(t5.shape) # output : [3, 4, 1]
print(t5)

#
t5 = t5.squeeze()
t4 = t4.squeeze()

# cat
t5 = t5.unsqueeze(dim=0)
t4 = t4.unsqueeze(dim=0)
t6 = torch.cat((t4, t5), dim=0) # depth 방향으로 행렬 병합
print(t6.shape) # output : [2, 3, 4]
print(t6)

#
t6 = torch.cat((t4, t5), dim=1) # 행 방향으로 행렬 병합
print(t6.shape) # output : [1, 6, 4]
print(t6)

# depth, 행, 열 중 어떤 것이 같아야 병합 가능한지
t6 = torch.cat((t4, t5), dim=2) # 열 방향으로 행렬 병합
print(t6.shape) # output : [1, 3, 8]
print(t6)

# stack와 cat의 차이
t6 = torch.stack((t2, t3), dim=0)
print(t6.shape) # output : [2, 3, 4]
print(t6)

t6 = torch.stack((t2, t3), dim=1)
print(t6.shape) # output : [3, 2, 4]
print(t6)

t6 = torch.stack((t2, t3), dim=2) # cat은 불가능
print(t6.shape) # output : [3, 4, 2]
print(t6)