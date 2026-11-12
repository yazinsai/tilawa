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
clean = dict(id='clean', speaker='training', split='train', kind='none', events=[], non_errors=[], scored=True)
assert f.training_labels([clean]) == [clean]
for rejected in [dict(excluded='audit'), dict(exclude=True), dict(scored=False), dict(unsupported=['mistake']),
                 dict(events=[{'errorClass': 'omitted_word'}]), dict(non_errors=['repeat']),
                 dict(split='eval'), dict(kind='skip_word'), dict(split=None), dict(kind=None)]:
    assert not f.training_labels([{**clean, **rejected}]), rejected
assert not f.training_labels([clean, {**clean, 'id': 'held', 'split': 'eval'}])
print('training selection rejects unresolved/excluded scripts and all held-out-speaker recordings')
