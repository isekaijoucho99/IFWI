import threading
import numpy as np
import scipy.sparse as sp
import pytest
from scipy.sparse.linalg import splu
from experiments.rom.owned_lu import OwnedLUGroup


def test_parallel_lu_is_destroyed_on_the_thread_that_created_it(monkeypatch):
    import experiments.rom.owned_lu as module
    events=[]
    class Factor:
        nnz=8
        def __init__(self,matrix):self.creator=threading.get_ident()
        def solve(self,rhs,trans='N'):return np.asarray(rhs).copy()
        def __del__(self):events.append((self.creator,threading.get_ident()))
    monkeypatch.setattr(module,'splu',Factor)
    group=OwnedLUGroup(2)
    factors=[group.factorize(k,sp.eye(3,format='csc')) for k in range(4)]
    group.close();group.close()
    assert len(events)==4 and all(created==destroyed for created,destroyed in events)
    assert all(created!=threading.get_ident() for created,destroyed in events)
    with pytest.raises(RuntimeError):factors[0].solve(np.ones(3))


def test_owned_native_solve_matches_serial_and_contains_backend_failures():
    matrix=sp.csc_matrix([[3+1j,1,0],[2,5-1j,1],[0,2,4+2j]])
    expected=splu(matrix);rhs=np.arange(3)+1j
    with OwnedLUGroup(2) as group:
        factor=group.factorize(0,matrix)
        for trans in ['N','T','H']:
            np.testing.assert_allclose(factor.solve(rhs,trans),expected.solve(rhs,trans),atol=1e-14)
        with pytest.raises(RuntimeError):group.factorize(1,sp.csc_matrix((3,3)))
        np.testing.assert_allclose(factor.solve(rhs),expected.solve(rhs),atol=1e-14)
