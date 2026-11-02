"""Keep native SuperLU lifetime on its allocating thread (SciPy1.15 TLS).

Only arrays, integers and serialized errors cross the owner boundary. Native
LU objects are never returned to a different thread for destruction.
"""
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse.linalg import splu


class _Owner:
    def __init__(self):
        self.executor=ThreadPoolExecutor(max_workers=1);self.factors={};self.closed=False
    def call(self,operation):
        if self.closed:raise RuntimeError('LU owner is closed')
        def guarded():
            try:return True,operation(),None
            except BaseException as error:
                return False,str(error),type(error).__name__
        okay,value,kind=self.executor.submit(guarded).result()
        if not okay:
            exception={'MemoryError':MemoryError,'ValueError':ValueError}.get(kind,RuntimeError)
            raise exception(kind+': '+value)
        return value
    def close(self):
        if not self.closed:
            self.call(self.factors.clear)
            self.closed=True;self.executor.shutdown(wait=True)


class _Factor:
    def __init__(self,owner,key,nnz):self.owner=owner;self.key=key;self.nnz=nnz
    def solve(self,rhs,trans='N'):
        return self.owner.call(lambda:self.owner.factors[self.key].solve(rhs,trans=trans))


class OwnedLUGroup:
    def __init__(self,workers):self.owners=[_Owner() for _ in range(workers)];self.closed=False
    def factorize(self,key,matrix):
        if self.closed:raise RuntimeError('LU group is closed')
        owner=self.owners[key%len(self.owners)]
        def build():
            owner.factors[key]=splu(matrix)
            return owner.factors[key].nnz
        return _Factor(owner,key,owner.call(build))
    def close(self):
        if not self.closed:
            for owner in self.owners:owner.close()
            self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def __del__(self):
        try:self.close()
        except Exception:pass
