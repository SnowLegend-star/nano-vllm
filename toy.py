import torch

# 1. 初始化原KVCache（2块，每块32slot，8个头，128维）
k_cache = torch.randn(2, 32, 8, 128)  # 随机值，方便对比
num_heads, head_dim = 8, 128

# 2. 扁平化（view共享内存）
k_cache_flat = k_cache.view(-1, num_heads, head_dim)

# 3. 修改flat的第0个slot为11
k_cache_flat[0] = 11
k_cache[0,1,0,0]=122
# 4. 验证：原k_cache的对应位置是否变成11
# 原k_cache的[0,0]位置 = flat的[0]位置（因为 0块×32slot + 0位置 = 第0个全局slot）
print("原k_cache[0,0]的第一个元素值：", k_cache[0, 0, 0, 0])  # 输出 11.0
print("flat[0]的第一个元素值：", k_cache_flat[0, 0, 0])       # 输出 11.0
print("flat[1]的第一个元素值：", k_cache_flat[1, 0, 0])     # idx=64

# 验证形状（和你代码一致）
print(k_cache.shape)       # torch.Size([2, 32, 8, 128])
print(k_cache_flat.shape)  # torch.Size([64, 8, 128])

# origin=torch.randn(2,4)
# print(origin)
# test=origin.chunk(2,-1)
# print(test)
# torch.arange()

x=torch.tensor([[1,2],[3,4]]) 
# x=x.float()
# var=x.pow(2).mean(0,True)
# print(var)

print(x.tolist())

