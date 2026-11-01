from pathlib import Path
import numpy as np
from experiments.rom.paper import ricker_laplace,weeks_inverse

FIXTURES=Path(__file__).resolve().parent/'fixtures'


def test_python_matches_actual_original_matlab_galerkin_export():
    data=np.load(FIXTURES/'upstream_matlab_v2_reference.npz')
    source=ricker_laplace(data['s_query'].ravel(),data['alpha'].item(),data['t0'].item(),data['amplitude'].item())
    np.testing.assert_allclose(source,data['q_query'].ravel(),rtol=1e-12,atol=1e-12)
    actual=np.stack([np.linalg.solve(s*s*data['M_r']+data['K_r'],data['b_r'].ravel()*q) for s,q in zip(data['s_query'].ravel(),source)],axis=1)
    np.testing.assert_allclose(actual,data['reduced_query'],rtol=2e-11,atol=1e-12)
    assert np.max(data['error_seismo_POD'][-1])<1e-4


def test_weeks_fft_clenshaw_matches_executed_original_matlab_function():
    data=np.load(FIXTURES/'upstream_core_fixture.npz')
    s=data['nodes'].ravel()
    response=np.stack([np.linalg.solve(z*z*data['M']+data['K'],data['b'].ravel()/(z+1.3)) for z in s])
    actual=weeks_inverse(response,data['time_grid'].ravel(),.2,3.)
    np.testing.assert_allclose(actual,data['seismogram'][0],atol=2e-13,rtol=2e-12)
