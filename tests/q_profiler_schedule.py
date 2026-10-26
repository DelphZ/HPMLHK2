OK_FORMAT = True

test = {
    'name': 'q_profiler_schedule',
    'points': 2,
    'suites': [{
        'cases': [{
            'code': r'''>>> def test_profiler_schedule(make_profiler_schedule):
...     from torch.profiler import ProfilerAction
...     sched = make_profiler_schedule()
...     actions = [sched(i) for i in range(8)]
...     record = (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE)
...     assert ProfilerAction.NONE in actions, 'schedule has no wait phase'
...     assert ProfilerAction.WARMUP in actions, 'schedule has no warmup phase'
...     assert any(a in record for a in actions), 'schedule never records'
...     first_warm = next(i for i, a in enumerate(actions) if a == ProfilerAction.WARMUP)
...     first_rec = next(i for i, a in enumerate(actions) if a in record)
...     assert first_warm < first_rec, 'warmup must precede the first recorded step'
>>> test_profiler_schedule(make_profiler_schedule)
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
