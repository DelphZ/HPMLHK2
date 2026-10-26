OK_FORMAT = True

test = {'name': 'q_optimizer_param_count',
 'points': 6,
 'suites': [{'cases': [{'code': '>>> def test_optimizer_param_count_matches(build_optimizer, '
                                'c6_model):\n'
                                '...     counts = {}\n'
                                "...     for name in ('sgd', 'adam', 'adamw'):\n"
                                '...         opt = build_optimizer(c6_model, name, lr=0.001)\n'
                                '...         counts[name] = sum((p.numel() for group in '
                                "opt.param_groups for p in group['params']))\n"
                                '...     total = sum((p.numel() for p in c6_model.parameters() if '
                                'p.requires_grad))\n'
                                "...     assert counts['sgd'] == counts['adam'] == counts['adamw'] "
                                '== total\n'
                                '>>> test_optimizer_param_count_matches(build_optimizer, '
                                'c6_model)\n',
                        'hidden': False,
                        'locked': False}],
             'scored': True,
             'setup': '',
             'teardown': '',
             'type': 'doctest'}]}
