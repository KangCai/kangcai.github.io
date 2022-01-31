
## 消掉最后一个 1

```pydocstring
def clearLastOne(int n):
    return n&(n-1)
```

### 扩展：计算二进制中 1 的个数

```pydocstring
def countOne(int n):
    int res = 0
    while n != 0:
        res += 1
        n &= n-1
    return res
```

### 扩展：消掉最后一个 0

```pydocstring
def clearLastZero(int n):
    n_ = ~n
    return ~(n&(n-1))
```

## 计算 n 的二进制表示中 1 数量的奇偶

```pydocstring

```