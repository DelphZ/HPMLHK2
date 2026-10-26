OK_FORMAT = True

test = {'name': 'q_train_step',
 'points': 3,
 'suites': [{'cases': [{'code': '>>> def test_train_step_runs(build_tiny_config, build_model, '
                                'collate_fn, tiny_rows, tokenize_function, tokenizer, '
                                'train_one_step):\n'
                                '...     torch.manual_seed(0)\n'
                                '...     model = build_model(build_tiny_config())\n'
                                '...     optimizer = torch.optim.AdamW(model.parameters(), '
                                'lr=0.01)\n'
                                "...     raw_batch = [{'input_ids': tokenize_function(r['text'], "
                                "tokenizer, max_len=64), 'label': r['label']} for r in "
                                'tiny_rows[:8]]\n'
                                '...     batch = collate_fn(raw_batch)\n'
                                '...     loss = train_one_step(model, batch, optimizer, '
                                "torch.device('cpu'))\n"
                                '...     assert isinstance(loss, float)\n'
                                '...     assert math.isfinite(loss)\n'
                                '...     assert loss > 0\n'
                                '>>> test_train_step_runs(build_tiny_config, build_model, '
                                'collate_fn, tiny_rows, tokenize_function, tokenizer, '
                                'train_one_step)\n',
                        'hidden': False,
                        'locked': False}],
             'scored': True,
             'setup': '',
             'teardown': '',
             'type': 'doctest'}]}
