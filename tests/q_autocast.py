OK_FORMAT = True

# Detection relies on the student calling torch.autocast(...) directly (as the
# handout instructs), not a name imported earlier as `from torch import autocast`.
test = {
    'name': 'q_autocast',
    'points': 3,
    'suites': [{
        'cases': [{
            'code': r'''>>> def test_autocast_used(build_tiny_config, build_model, collate_fn, tiny_rows, tokenize_function, tokenizer, mixed_precision_step):
...     torch.manual_seed(0)
...     seen = {'autocast': False}
...     orig_autocast = torch.autocast
...     def spy(*args, **kwargs):
...         seen['autocast'] = True
...         return orig_autocast(*args, **kwargs)
...     torch.autocast = spy
...     try:
...         model = build_model(build_tiny_config())
...         optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
...         raw_batch = [{'input_ids': tokenize_function(r['text'], tokenizer, max_len=64), 'label': r['label']} for r in tiny_rows[:8]]
...         batch = collate_fn(raw_batch)
...         mixed_precision_step(model, batch, optimizer, torch.device('cpu'))
...     finally:
...         torch.autocast = orig_autocast
...     assert seen['autocast'], 'mixed_precision_step did not call torch.autocast'
>>> test_autocast_used(build_tiny_config, build_model, collate_fn, tiny_rows, tokenize_function, tokenizer, mixed_precision_step)
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
