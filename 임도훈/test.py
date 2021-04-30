a={1,2,3,4}
b={2,3}
print(set(a)-(set(b)))
c=[]
c.extend((set(a)-set(b)))
print(c)
c.extend([1,2])
print(c)