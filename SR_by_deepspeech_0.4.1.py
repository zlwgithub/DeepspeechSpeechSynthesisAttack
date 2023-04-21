import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import scipy.io.wavfile as wav

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from tf_logits import get_logits

toks = " abcdefghijklmnopqrstuvwxyz'-"

def main():
    with tf.Session() as sess:

        _, audio = wav.read('example_l2ctc_helloworld.wav')

        N = len(audio)
        new_input = tf.placeholder(tf.float32, [1, N])
        lengths = tf.placeholder(tf.int32, [1])

        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            logits = get_logits(new_input, lengths)

        saver = tf.train.Saver()
        saver.restore(sess,  './deepspeech-0.4.1-checkpoint/model.v0.4.1')

        decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=500)

        length = (len(audio)-1)//320
        l = len(audio)
        r = sess.run(decoded, {new_input: [audio],
                               lengths: [length]})

        # print(r[0][1])

        print("-"*80)
        print("-"*80)

        print("Classification:")
        print("".join([toks[x] for x in r[0].values]))
        print("-"*80)
        print("-"*80)

main()
