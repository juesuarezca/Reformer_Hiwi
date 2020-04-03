import gin
import os
import jax
import trax
from trax.supervised import inputs
from trax.shapes import ShapeDtype
import numpy as onp
import jax.numpy as np

from scipy.special import softmax

from sentencepiece import SentencePieceProcessor

inpt1=np.array([[1,1,1,1,1,1,1,1]])
inpt=(inpt1,inpt1)
vocab_size=16
model=trax.models.reformer.Reformer(
    vocab_size,output_vocab_size=vocab_size, d_model=32, d_ff=64,max_len=34,mode='train')
vocab_size=16
def refmodel(mode):
    return trax.models.reformer.Reformer1(
    vocab_size,output_vocab_size=vocab_size, d_model=32, d_ff=64,max_len=16,mode=mode)
import numpy
def copy_task(batch_size, vocab_size, length):
    while True:
        w_length=7
        loss_weights = onp.concatenate([onp.zeros((batch_size, w_length)),
                                       onp.ones((batch_size, w_length+2))], axis=1)
        zero = np.zeros([batch_size, 1], np.int32)
        w = onp.ones([batch_size, vocab_size], onp.int32)
        x=np.concatenate([zero, w, zero,w], axis=1)
        inputs=x#[l.tolist() for l in numpy.array(x)]
        outputs=x
        print(inputs.shape)
        yield (inputs, outputs, inputs)#(inputs, outputs, loss_weights)
        
copy_inputs = trax.supervised.Inputs(lambda _: copy_task(10, 16, 8))

output_dir = os.path.expanduser('~/train_dir/')
!rm -f ~/train_dir/model.pkl  # Remove old model
trainer = trax.supervised.Trainer(
    model=refmodel,#tiny_transformer_lm,
    loss_fn=trax.layers.CrossEntropyLoss,
    optimizer=trax.optimizers.Adam,
    lr_schedule=trax.lr.MultifactorSchedule,
    inputs=copy_inputs,
    output_dir=output_dir,
    has_weights=True)
print('ready')
