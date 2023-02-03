# feature transformations (e.g. embedding net)

def identity(x):
    return x

def first_dim_only(x):
    return x[:,0].reshape(-1,1)