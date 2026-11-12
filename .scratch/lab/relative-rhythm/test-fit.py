"""Small math checks for the estimator; execute on Modal along with the TypeScript tests."""
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location('rhythm_fit', 'packages/core/src/rhythm-fit.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
q, pair = f.correspondence('بَتننن', list('بَن'))
assert np.allclose(q.sum(axis=1), 1)
assert np.allclose(pair.sum(axis=(1, 2)), 1)
assert np.allclose(pair.sum(axis=2), q[:-1])
assert np.allclose(pair.sum(axis=1), q[1:])
assert pair[1, :, 4].sum() > 0
print('soft correspondence marginals normalized and consistent')
