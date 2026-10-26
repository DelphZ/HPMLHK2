OK_FORMAT = True

# Checks that gb_before and gb_after are well-formed integers and that your
# fix reduced the graph-break count.
test = {
    'name': 'q_graph_break_counts',
    'points': 2,
    'suites': [{
        'cases': [{
            'code': r'''>>> def test_graph_break_counts(gb_before, gb_after):
...     assert isinstance(gb_before, int) and isinstance(gb_after, int)
...     assert gb_before >= 0 and gb_after >= 0
...     assert gb_after < gb_before, 'graph-break count should drop after your fix'
>>> test_graph_break_counts(gb_before, gb_after)
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
