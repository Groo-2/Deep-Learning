import torch
import numpy as np

# 이 컴퓨터는 GPU 사용 가능능
device = "cuda" if torch.cuda.is_available() else "CPU"
print(device)

# cuda:0 : 첫번째 GPU에 해당당
t1 = torch.tensor([1,2,3], dtype=torch.float16, requires_grad=False, device=device)
print(t1)

# t1의 속성
print(type(t1))
print(t1.requires_grad)

# tensor 값 수정
t1[0] = 100
print(t1)

# 리스트 -> 배열로 전환
arr = [1,2,3,4]
n1 = np.array(arr, dtype=np.float32)
print(type(n1)) # 타입 : ndarray
print(n1) # n1 출력 : 1., 2., 3., 4.
print(n1.dtype) # float32
print(n1.shape) # 크기

#
t1 = torch.tensor(arr, dtype=torch.float32, device=device)
print(type(t1)) # torch.Tensor
print(t1)
print(t1.requires_grad) # default : False
print(t1.shape) # 크기

# numpy 구조체 -> torch.Tensor 구조체
t2 = torch.from_numpy(n1)
print(type(t2))
print(t2)

#
t2[1] = 100
print(t2) # 1., 100., 3., 4.

# tensor -> numpy
n2 = t2.numpy()
print(type(n2)) # 타입 : ndarray

#
t2 = torch.from_numpy(n1).to(device=device) # t2 = torch.from_numpy(n1).cuda()
print(type(t2)) # 타입 : Tensor
print(t2.device) # cuda:0
print(t2.dtype) # torch.float32
print(t2.requires_grad) # False(default)

#
t3 = torch.Tensor(arr)
print(type(t3)) # 타입 : Tensor
print(t3) # 1., 2., 3., 4.

# torch로 일(1) 행렬 만들기
t1 = torch.ones((3,4), device=device, dtype=torch.float32, requires_grad=False)
print(type(t1)) # 타입 : torch.Tensor
print(t1) # 3x4 일(1) 행렬

# numpy로 일(1) 행렬 만들기
n1 = np.ones((3,4), dtype=np.float32)
print(type(n1)) # 타입 : numpy.ndarray
t2 = torch.from_numpy(n1).to(dtype=torch.float16) # torch의 데이터 타입을 넣어줘야 됨
print(t2)

# 랜덤 행렬
n1 = np.random.rand(3,3) # dtype = np.float16 안됨됨
print(type(n1))
print(n1)
t1 = torch.rand(3,3)
print(type(t1)) # 타입 : Tensor
print(t1)

# 동일한 딥러닝 실험 결과를 얻기 위해 랜덤 seed 사용
torch.manual_seed(0) # 정수값 입력
t1 = torch.rand(4)
print(t1)
t2 = torch.rand(4)
print(t2)
torch.maual_seed(0)
t1 = torch.rand(4)
t2 = torch.rand(4)
print(t1)
print(t2)