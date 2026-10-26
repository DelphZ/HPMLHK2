OK_FORMAT = True

test = {'name': 'q3_param_counts',
 'points': 5,
 'suites': [{'cases': [{'code': '>>> def test_param_counts_basic(q3_model, n_trainable, '
                                'n_with_grad_before, n_with_grad_after):\n'
                                '...     assert n_with_grad_before == 0\n'
                                '...     assert n_trainable > 0\n'
                                '...     assert n_with_grad_after == n_trainable\n'
                                '...     assert n_trainable == sum((p.numel() for p in '
                                'q3_model.parameters()))\n'
                                '>>> test_param_counts_basic(q3_model, n_trainable, '
                                'n_with_grad_before, n_with_grad_after)\n',
                        'hidden': False,
                        'locked': False}],
             'scored': True,
             'setup': '',
             'teardown': '',
             'type': 'doctest'}]}
