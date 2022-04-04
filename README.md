At their core, neural networks are remarkably simple. Even modern, state-of-the-art architectures such as the transformer can be [described mathematically in about half a page of A4 paper](https://johnthickstun.com/docs/transformers.pdf).

Given this, one would be forgiven for assuming that practical implementations of neural nets would be easy to understand and extend. As anyone who's tried to read the code can tell you, this is not the case. Implementations are long and complex. Rarely is there a clear visual distinction between 

Convenience functions, designed to make code easier to use, are like landmines of bloat placed there to waste the reader's time. Libraries are a tangled web of files and classes, with boilerplate for initializing variables and passing data around. Look at this code. This is not the writer's fault; this good code by a good writer.

```python
class TransformerModel(tf.keras.layers.Layer):
  def __init__(self, 
               vocab_size,
               encoder_stack_size=6, 
               decoder_stack_size=6, 
               hidden_size=512, 
               num_heads=8, 
               filter_size=2048, 
               dropout_rate=0.1,
               extra_decode_length=50,
               beam_width=4,
               alpha=0.6):
    super(TransformerModel, self).__init__()
    self._vocab_size = vocab_size
    self._encoder_stack_size = encoder_stack_size
    self._decoder_stack_size = decoder_stack_size
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._filter_size = filter_size
    self._dropout_rate = dropout_rate
    self._extra_decode_length = extra_decode_length
    self._beam_width = beam_width
    self._alpha = alpha
```

