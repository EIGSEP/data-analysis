"""Verify that the five completed chains used one unchanged UTM target."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

import make_product as product


def hash_array(value):
    return hashlib.sha256(np.asarray(value, dtype=float).tobytes()).hexdigest()


def main(out=None):
    product.verify()
    _, model = product.load_model()
    with np.load(product.HERE / 'inputs/starting_states.npz') as starts:
        initial = starts['states']
    signatures = []
    rows = []
    for index, chain_id in enumerate(product.CHAINS):
        checkpoint = product.HERE / 'run' / f'checkpoint_{chain_id}.pkl'
        # The checkpoint is produced locally by this run and is not accepted
        # from a remote source. Pickle is needed for exact RNG state recovery.
        with checkpoint.open('rb') as stream:
            saved = pickle.load(stream)
        signature = saved['signature']
        assert saved['next_iteration'] == 2300
        assert signature['start_sha256'] == hash_array(initial[index])
        assert (signature['tune'], signature['draws']) == (300, 2000)
        assert signature['thin_points'] == 10
        assert signature['joint_directions_sha256'] is None
        assert signature['difference_step_factor'] == .1
        assert signature['config'] == product.verify()['config']
        # Recorded as absolute paths on the run host, some under terrain/.
        assert all(product.sha(product.resolve(path)) == digest
                   for path, digest in signature['inputs'].items())
        output = product.HERE / 'run' / f'chain_{chain_id}.npz'
        with np.load(output) as result:
            assert result['globals'].shape == (2000, model.ng)
            assert result['logp'].shape == (200, 2)
            assert result['landmarks'].shape == (200, model.npoint, 3)
            np.testing.assert_equal(result['start'], initial[index, :model.ng])
            np.testing.assert_equal(result['globals'], saved['records'])
            np.testing.assert_equal(result['logp'], saved['stats'])
            np.testing.assert_equal(result['landmarks'], saved['landmarks'])
            chain = saved['chain']
            final = model.pack(chain['cam'], chain['ant'], chain['tx'],
                               chain['bias'], chain['extra'], chain['tx_extra'],
                               chain['points'])
            np.testing.assert_equal(result['final'], final)
        signatures.append(signature['inputs'])
        rows.append(dict(chain=chain_id, checkpoint_sha256=product.sha(checkpoint),
                         result_sha256=product.sha(output),
                         final_log_density=float(model.logp(final))))
    assert all(value == signatures[0] for value in signatures[1:])
    report = dict(status='pass', scope='five completed 300+2000 UTM-target chains',
                  input_sha256=signatures[0], chains=rows)
    out = Path(out) if out else product.HERE / 'run_integrity.json'
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', help='write the report here instead of the product '
                        '(to compare against the published run_integrity.json)')
    main(parser.parse_args().out)
