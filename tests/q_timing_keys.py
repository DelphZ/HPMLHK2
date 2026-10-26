OK_FORMAT = True

test = {
    'name': 'q_timing_keys',
    'points': 3,
    'suites': [{
        'cases': [{
            'code': r'''>>> def test_timing_keys(build_tiny_config, build_model, collate_fn, tiny_rows, tokenize_function, tokenizer, time_training_step):
...     torch.manual_seed(0)
...     model = build_model(build_tiny_config())
...     optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
...     raw_batch = [{'input_ids': tokenize_function(r['text'], tokenizer, max_len=64), 'label': r['label']} for r in tiny_rows[:8]]
...     batch = collate_fn(raw_batch)
...     t = time_training_step(model, batch, optimizer, torch.device('cpu'))
...     required = {'data', 'forward', 'backward', 'optimizer', 'total'}
...     assert required.issubset(set(t.keys()))
...     for k in ('forward', 'backward', 'optimizer', 'total'):
...         assert isinstance(t[k], float) and t[k] >= 0.0
>>> test_timing_keys(build_tiny_config, build_model, collate_fn, tiny_rows, tokenize_function, tokenizer, time_training_step)
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
