OK_FORMAT = True

test = {'name': 'q1_embedding_dim',
 'points': 3,
 'suites': [{'cases': [{'code': '>>> def test_embedding_dim_matches_vocab(tokenizer, '
                                'embedding_input_dim):\n'
                                '...     assert embedding_input_dim == tokenizer.vocab_size\n'
                                '>>> test_embedding_dim_matches_vocab(tokenizer, '
                                'embedding_input_dim)\n',
                        'hidden': False,
                        'locked': False,
                        'points': 1}],
             'scored': True,
             'setup': '',
             'teardown': '',
             'type': 'doctest'}]}
