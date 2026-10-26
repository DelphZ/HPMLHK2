OK_FORMAT = True

test = {'name': 'q_collate',
 'points': 3,
 'suites': [{'cases': [{'code': '>>> def test_collate_shapes(tokenizer, tokenize_function, '
                                'make_collate_fn):\n'
                                "...     texts = ['good movie', 'this was a terrible and boring "
                                "film honestly', 'ok']\n"
                                '...     labels = [1, 0, 1]\n'
                                "...     batch = [{'input_ids': tokenize_function(t, tokenizer, "
                                "max_len=64), 'label': l} for (t, l) in zip(texts, labels)]\n"
                                '...     collate = make_collate_fn(tokenizer)\n'
                                '...     out = collate(batch)\n'
                                "...     assert set(out.keys()) == {'input_ids', 'attention_mask', "
                                "'labels'}\n"
                                "...     max_len = max((len(ex['input_ids']) for ex in batch))\n"
                                "...     assert out['input_ids'].shape == (3, max_len)\n"
                                "...     assert out['attention_mask'].shape == (3, max_len)\n"
                                "...     assert out['labels'].tolist() == labels\n"
                                '...     longest_idx = max(range(3), key=lambda i: '
                                "len(batch[i]['input_ids']))\n"
                                "...     assert out['attention_mask'][longest_idx].sum().item() == "
                                'max_len\n'
                                '>>> test_collate_shapes(tokenizer, tokenize_function, '
                                'make_collate_fn)\n',
                        'hidden': False,
                        'locked': False}],
             'scored': True,
             'setup': '',
             'teardown': '',
             'type': 'doctest'}]}
