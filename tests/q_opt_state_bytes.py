OK_FORMAT = True

test = {
    'name': 'q_opt_state_bytes',
    'points': 2,
    'suites': [{
        'cases': [{
            'code': r'''>>> def test_opt_state_bytes(optimizer_state_bytes):
...     # fp32 optimizer state per parameter: SGD none, SGD+momentum one buffer,
...     # Adam / AdamW two moment buffers (4 bytes each at fp32).
...     assert optimizer_state_bytes(1000, 'sgd') == 0
...     assert optimizer_state_bytes(1000, 'sgd', momentum=0.9) == 4000
...     assert optimizer_state_bytes(1000, 'adam') == 8000
...     assert optimizer_state_bytes(1000, 'adamw') == 8000
...     assert isinstance(optimizer_state_bytes(123, 'adamw'), int)
>>> test_opt_state_bytes(optimizer_state_bytes)
''',
            'hidden': False,
            'locked': False,
        }],
        'scored': True,
        'setup': '',
        'teardown': '',
        'type': 'doctest',
    }],
}
