import torch
import numpy as np

#
device = "cuda" if torch.cuda.is_available() else "cpu"
t1 = torch.tensor([1,2,3,4], dtype=torch.float64, device=device, requires_grad=False)
print(t1)

# 일(1) 행렬렬
t1 = torch.ones((3,3), dtype=torch.float32, device=device, requires_grad=True)
print(t1)
print(t1.shape) # print(t1.size())

# 영(0) 행렬렬
t2 = torch.zeros((3,4)) # default 타입, default device = cpu
print(t2.shape)
print(t2.device)
print(t2.requires_grad)

# linspace
t1 = torch.linspace(1, 10, 10) # 1~10까지 10으로 쪼개서
print(t1) # 10포함함
print(t1.shape) # output : 10

# arange
t2 = torch.arange(1, 10, 1) # 1~10까지 1씩 증가
print(t2) # 10제외함
print(t2.shape) # output : 9

# eye 행렬
t3 = torch.eye(4, dtype=torch.float16)
print(t3)

# view
t2 = t1.view(3,4) # 크기는 바뀌면 안되고 차원만 바뀌어야 함
print(t2)

# reshape
t2 = t1.reshape(3,4)
print(t2)

# transpose 전치행렬 : 메모리 재배치 x, 인덱스 재순서 o
t3 = t2.t()
print(t3)

# contiguous # 순서는 바뀌지 않고 인덱스 값만 바뀜
print(t3.is_contiguous()) # output : False
print(t2.is_contiguous()) # output : True
t4 = t3.view(3,4) # output : Error -> contiguous 하지 않기 때문에
t4 = t3.reshape(3,4) # contiguous하지 않더라도 변환 가능
print(t4) # 순서가 바뀜

# view를 reshape처럼 쓰려면
t3 = t3.contiguous()
print(t3.is_contiguous()) # output : True
t3 = t3.view(3,4)
print(t3) # output : reshape과 같은 결과

#
t3 = t2.t()
t4 = t3.t()

# transpose
t3 = t2.transpose(0, 1) # 0번 축과 1번 축을 서로 바꿈

# clone # 똑같은 것을 다른 메모리에 할당
t3 = t2.clone() # Deep Copy
t3[1,1] = 100
print(t2) # t2에 영향을 주지 않는 것을 볼 수 있음
