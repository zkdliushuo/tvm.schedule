import tvm

n = 1024
A = tvm.placeholder((n,), name='A')         # 定义A， vertor size = 1024
k = tvm.reduce_axis((0, n), name='k')       # 规定reduction操作的轴 

B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k), name='B')    # 定义compute，对A求和存入B

s = tvm.create_schedule(B.op)               # 定义schedule

print(tvm.lower(s, [A, B], simple_mode=True))   # 生成
print("---------cutting line---------")

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)   # 切割

print(tvm.lower(s, [A, B], simple_mode=True))   # 生成
