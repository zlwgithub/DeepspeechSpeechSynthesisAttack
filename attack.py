import numpy as np
import scipy.io.wavfile as wav
import time
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits
import math
toks = " abcdefghijklmnopqrstuvwxyz'-"

class Attack:
    def __init__(self, sess, phrase_length, max_audio_len,
                 learning_rate, num_iterations, batch_size,
                 restore_path):
        self.sess = sess
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len
        # self.bird  = bird

        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name='qq_phrase_lengths')


        self.apply_delta = tf.clip_by_value(delta, -2000, 2000)
        self.new_input = new_input = self.apply_delta+ original


        #加一个噪音减少过拟合
        noise = tf.random_normal(new_input.shape,stddev=2)
        pass_in = tf.clip_by_value(new_input+noise, -2**15, 2**15-1)

        self.logits = logits = get_logits(pass_in, lengths)

        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, restore_path)


        target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths)

        # dbloss = 20*tf.math.log(tf.reduce_max(tf.abs(self.apply_delta)))/tf.math.log(10.)
        self.ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),inputs=logits, sequence_length=lengths)
        # noiseloss =
        # self.birdloss = tf.reduce_mean(tf.square(self.apply_delta-self.bird))

        self.l2loss =tf.reduce_mean(tf.square(self.apply_delta))
     
        self.loss =  self.ctcloss
        

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(learning_rate)

        grad,var = optimizer.compute_gradients(self.loss, [delta])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad),var)])
        
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        
        sess.run(tf.variables_initializer(new_vars+[delta]))


        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=100)
        # print(self.decoded)
        # print(type(self.decoded))

    def attack(self, audio, lengths, target, finetune=None):
        global res
        sess = self.sess
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        sess.run(self.lengths.assign((np.array(lengths)-1)//320))
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths)-1)//320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))

        final_deltas = [None]*self.batch_size

        if finetune is not None and len(finetune) > 0:
            sess.run(self.delta.assign(finetune-audio))

        now = time.time()
        MAX = self.num_iterations
        for i in range(MAX):
            iteration = i
            now = time.time()


            if i%10 == 0:
                new, delta, r_out, r_logits = sess.run((self.new_input, self.delta, self.decoded, self.logits))
                lst = [(r_out, r_logits)]


                for out, logits in lst:
                    chars = out[0].values

                    res = np.zeros(out[0].dense_shape)+len(toks)-1

                    for ii in range(len(out[0].values)):
                        x,y = out[0].indices[ii]
                        res[x,y] = out[0].values[ii]

                    # Here we print the strings that are recognized.
                    res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                    print("\n".join(res))

                    # And here we print the argmax of the alignment.
                    res2 = np.argmax(logits,axis=2).T
                    res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,lengths)]
                    print("\n".join(res2))
            feed_dict = {}
            ctc,d, l, logits, new_input, _ = sess.run((self.ctcloss,self.delta,
                                                           self.loss,
                                                           self.logits, self.new_input,
                                                           self.train),
                                                          feed_dict)

            # Report progress
            print("loss   %.3f" % np.mean(l), "\t", "\t".join("%.3f" % x for x in l))
            # print("birdloss  %.3f" % np.mean(bl), "\t", "\t".join("%.3f" % x for x in l))
            # print("l2loss   %.3f" % np.mean(ll), "\t", "\t".join("%.3f" % x for x in l))
            # print("ctcloss   %.3f" % np.mean(ctc), "\t", "\t".join("%.3f" % x for x in l))

            logits = np.argmax(logits, axis=2).T
            # Every 100 iterations, check if we've succeeded
            # if we have (or if it's the final epoch) then we
            # should record our progress and decrease the
            # rescale constant.
            if (i % 10 == 0 and res[0] == "".join([toks[x] for x in target[0]]))or (i == MAX - 1 and final_deltas[0] is None):
                # # Get the current constant
                # rescale = sess.run(self.rescale)
                # if rescale[0] * 2000 > np.max(np.abs(d)):
                #     # If we're already below the threshold, then
                #     # just reduce the threshold to the current
                #     # point and save some time.
                #     print("It's way over", np.max(np.abs(d[0])) / 2000.0)
                #     rescale[0] = np.max(np.abs(d[0])) / 2000.0
                #
                # # Otherwise reduce it by some constant. The closer
                # # this number is to 1, the better quality the result
                # # will be. The smaller, the quicker we'll converge
                # # on a result but it will be lower quality.
                # rescale[0] *= .8

                # Adjust the best solution found so far
                final_deltas[0] = new_input[0]

                # print("Worked i=%d ctcloss=%f bound=%f" % (0, l[0], 2000 * rescale[0][0]))
                # # print('delta',np.max(np.abs(new_input[ii]-audio[ii])))
                # sess.run(self.rescale.assign(rescale))
                break

                # Just for debugging, save the adversarial example
                # to /tmp so we can see it if we want

        return final_deltas
    
def main():
    with tf.Session() as sess:
        finetune = []
        audios = []
        lengths = []


        fs, audio = wav.read('./original/t3.wav')
        assert fs == 16000
        assert audio.dtype == np.int16
        print('source dB', 20*np.log10(np.max(np.abs(audio))))
        audios.append(list(audio))
        lengths.append(len(audio))


        finetune.append(list(wav.read('./original/t3.wav')[1]))

        maxlen = max(map(len,audios))
        audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
        finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])
        # rate_bird, sig_bird = wav.read('bird.wav')
        # sig_bird = sig_bird / 10
        phrase = "hello world"
        # len_delt = len(audio)
        # len_bird = len(sig_bird)
        # if len_delt>len_bird:
        #     bird = np.tile(sig_bird,math.floor(len_bird/len_bird)+1)
        #     bird = bird[:len_delt]
        # else:
        #     bird = sig_bird[:len_delt]

        # Set up the attack class and run it
        attack = Attack(sess, len(phrase), maxlen,
                        batch_size=len(audios),
                        learning_rate=1,
                        num_iterations=100000000000,
                        restore_path='./deepspeech-0.4.1-checkpoint/model.v0.4.1')
        deltas = attack.attack(audios,
                               lengths,
                               [[toks.index(x) for x in phrase]]*len(audios),
                               finetune)

        for i in range(1):
            wav.write("example_max.wav", 16000,np.array(np.clip(np.round(deltas[i][:lengths[i]]),-2**15, 2**15-1),dtype=np.int16))
            print("Final distortion", np.max(np.abs(deltas[i][:lengths[i]]-audios[i][:lengths[i]])))
main()
