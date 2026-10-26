OK_FORMAT = True

test = {
    'name': 'q_record_function',
    'points': 2,
    'suites': [{
        'cases': [{
            'code': r'''>>> def test_record_function_labels(build_tiny_config, build_model, collate_fn, tiny_rows, tokenize_function, tokenizer, profiled_step):
...     from torch.profiler import profile, ProfilerActivity
...     torch.manual_seed(0)
...     model = build_model(build_tiny_config())
...     optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
...     raw_batch = [{'input_ids': tokenize_function(r['text'], tokenizer, max_len=64), 'label': r['label']} for r in tiny_rows[:8]]
...     batch = collate_fn(raw_batch)
...     with profile(activities=[ProfilerActivity.CPU]) as prof:
...         profiled_step(model, batch, optimizer, torch.device('cpu'))
...     names = {evt.key for evt in prof.key_averages()}
...     for label in ('forward', 'backward', 'optimizer'):
...         assert any(label in n for n in names), 'missing record_function label: ' + label
>>> test_record_function_labels(build_tiny_config, build_model, collate_fn, tiny_rows, tokenize_function, tokenizer, profiled_step)
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
