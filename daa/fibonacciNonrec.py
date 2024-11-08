def fibonacci(n):
    if n <= 0:
        return []
    
    if n  == 1:
        return [0]
    
    if n == 2:
        return [0,1]
    
    fib_seq = [0,1]

    for i in range(2,n):
        next_num = fib_seq[i-1] + fib_seq[i-2]
        fib_seq.append(next_num)
    
    return fib_seq

n = 10
print(fibonacci(n))
    


