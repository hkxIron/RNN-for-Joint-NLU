# coding=utf-8
# @author: cer
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys


class Model:
    def __init__(self, input_steps, embedding_size, hidden_size, vocab_size, slot_size,
                 intent_size, epoch_num, batch_size=16, n_layers=1):
        # input_steps:输入的序列长度
        self.input_steps = input_steps
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.epoch_num = epoch_num
        # encoder_inputs:[encoder_sequence_length, batch]
        self.encoder_inputs = tf.placeholder(tf.int32, [input_steps, batch_size],
                                             name='encoder_inputs')
        # 每句输入的实际长度，除了padding
        # encoder_inputs_actual_length:[batch]
        self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size],
                                                           name='encoder_inputs_actual_length')
        # decoder_targets:[batch, decoder_sequence_length]
        self.decoder_targets = tf.placeholder(tf.int32, [batch_size, input_steps],
                                              name='decoder_targets')
        # intent_targets:[batch]
        self.intent_targets = tf.placeholder(tf.int32, [batch_size],
                                             name='intent_targets')

    def build(self):
        # embeddings:[vocab_size, embedding_size]
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size],
                                                        -0.1, 0.1), dtype=tf.float32, name="embedding")
        # encoder_inputs_embedd: [encoder_sequence_length,batch, embedding_size]
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        # Encoder

        # 使用单个LSTM cell
        encoder_f_cell_0 = LSTMCell(self.hidden_size) # 这里只使用了单层的rnn
        encoder_b_cell_0 = LSTMCell(self.hidden_size)
        encoder_f_cell = DropoutWrapper(encoder_f_cell_0, output_keep_prob=0.5)
        encoder_b_cell = DropoutWrapper(encoder_b_cell_0, output_keep_prob=0.5)

        # encoder_inputs_time_major = tf.transpose(self.encoder_inputs_embedded, perm=[1, 0, 2])
        # 下面四个变量的尺寸：T*B*D，T*B*D，B*D，B*D

        #(output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
        # output_fw: [batch, input_sequence_length, num_units],它的值为hidden_state
        # output_bw: [batch, input_sequence_length, num_units],它的值为hidden_state
        # (cell_state_fw, hidden_state_fw) = states_fw
        # cell_state_fw: [batch, num_units]
        # hidden_state_fw: [batch, num_units]
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_f_cell,
                                            cell_bw=encoder_b_cell,
                                            inputs=self.encoder_inputs_embedded,
                                            sequence_length=self.encoder_inputs_actual_length,
                                            dtype=tf.float32,
                                            time_major=True) # time_major为主的速度会快一些
        # encoder_outputs:[encoder_sequence_length, batch, 2*hidden_size]
        encoder_outputs = tf.concat(values=(encoder_fw_outputs, encoder_bw_outputs), axis=2)

        # cell_state concat
        # encoder_final_c:[batch, hidden_size*2]
        encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), axis=1)

        # encoder_final_state_h:[batch, hidden_size*2]
        encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), axis=1)

        # 组成新的hidden_state
        self.encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )
        print("encoder_outputs: ", encoder_outputs)
        print("encoder_outputs[0]: ", encoder_outputs[0])
        print("encoder_final_state_c: ", encoder_final_state_c)

        # Decoder
        # decoder_lengths:[batch]
        decoder_lengths = self.encoder_inputs_actual_length
        # slot_W: [hidden_size*2, slot_size]
        self.slot_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.slot_size], -1, 1),
                             dtype=tf.float32, name="slot_W")
        # slot_bias:[slot_size]
        self.slot_b = tf.Variable(tf.zeros([self.slot_size]), dtype=tf.float32, name="slot_b")

        # intent_W:[hidden_size*2, intent_size]
        intent_W = tf.Variable(tf.random_uniform([self.hidden_size * 2, self.intent_size], -0.1, 0.1),
                               dtype=tf.float32, name="intent_W")
        # intent_b:[intent_size]
        intent_b = tf.Variable(tf.zeros([self.intent_size]), dtype=tf.float32, name="intent_b")

        # 求intent
        # encoder_final_state_h:[batch, hidden_size*2]
        # intent_W:[hidden_size*2, intent_size]
        # intent_logits:[batch, intent_size]
        intent_logits = tf.add(tf.matmul(encoder_final_state_h, intent_W), intent_b)
        # intent_prob:[batch]
        intent_prob = tf.nn.softmax(intent_logits)
        # intent:[batch]
        self.intent = tf.argmax(intent_logits, axis=1)

        # TODO: sos我看index为2
        # sos_time_slice:[batch]
        sos_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='SOS') * 2 # 注意:sos index为2
        # sos_step_embedded:[batch, embedding_size]
        sos_step_embedded = tf.nn.embedding_lookup(self.embeddings, sos_time_slice)
        # pad_time_slice = tf.zeros([self.batch_size], dtype=tf.int32, name='PAD')
        # pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, pad_time_slice)
        # pad_step_embedded:[batch, hidden_size*2+embedding_size]
        pad_step_embedded = tf.zeros([self.batch_size, self.hidden_size*2+self.embedding_size],
                                     dtype=tf.float32)

        # callable that returns (finished, next_inputs) for the first iteration
        # return finished:是否时间步结束
        #       initial_input:初始decoder rnn的输入
        def initial_fn():
            # initial_elements_finished:[batch]
            initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
            # encoder_outputs:[encoder_sequence_length, batch, 2*hidden_size]
            # last_hidden_state = [batch, 2*hidden_size]
            last_hidden_state = encoder_outputs[0]
            # decoder_input = [batch, embedding_size]
            decoder_input = sos_step_embedded
            # initial_input:[batch, 2*hidden_size+embedding_size]
            initial_input = tf.concat((decoder_input, last_hidden_state), axis=1)
            return initial_elements_finished, initial_input

        """
        decoder的output是每个词的概率,具体要生成哪个词,可以采用不同的策略.
        直接选择概率最大的,就是greedy,还可以用beam search的策略,可以了解下。
        decoder中把每一个时刻产生的输出重新embedding，作为下一个时刻的输入之一
        """
        # callable that takes (time, outputs, state) and emits tensor sample_ids.
        def sample_fn(time, outputs, state):
            # 选择logit最大的下标作为sample
            # outputs:[batch, slot_size]
            print("sample_fn outputs:", outputs)
            # output_logits = tf.add(tf.matmul(outputs, self.slot_W), self.slot_b)
            # print("slot output_logits: ", output_logits)
            # prediction_id = tf.argmax(output_logits, axis=1)

            # 注意: 这里的output已经是 decoder的输出乘以W project后的结果了
            # outputs:[batch, slot_size]
            # prediction_id:[batch]
            prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
            return prediction_id

        # callable that takes (time, outputs, state, sample_ids) and emits (finished, next_inputs, next_state).
        def next_inputs_fn(time, outputs, state, sample_ids):
            sample_ids_of_last_step = sample_ids
            # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
            # outputs:[batch, slot_size]
            print("next_inputs_fn outputs:", outputs)
            # sample_ids_of_last_step:[batch]
            # pred_embedding:[batch, embedding_size]
            last_step_label_embedding = tf.nn.embedding_lookup(self.embeddings, sample_ids_of_last_step)
            # encoder_outputs:[encoder_sequence_length, batch, 2*hidden_size]
            # next_input:[batch, embedding_size+2*hidden_size]
            # 输入是: h_i+o_{i-1}+c_i, 好像此处没有c_i, c_i在下面的Attention中才加上
            # s_i=f(s_{i-1}, y_{i-1}, h_i,c_i), h_i:为encoder时的hidden
            next_input = tf.concat((last_step_label_embedding, encoder_outputs[time]), axis=1)
            # elements_finished:[batch]
            elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
            # "logical and" across all dimensions
            all_finished = tf.reduce_all(elements_finished)  # -> boolean scalar
            # next_input:[batch, embedding_size+2*hidden_size]
            next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
            # next_state:AttentionWrapperState(cell_state=LSTMStateTuple(cell,hidden,attention,align,...))
            next_state = state
            print("next_inputs_fn elements_finished:", elements_finished)
            print("next_inputs_fn next_inputs:", next_inputs)
            print("next_inputs_fn next_state:", next_state)
            return elements_finished, next_inputs, next_state
        # helper一般用来定义decoder input的输入
        my_helper = tf.contrib.seq2seq.CustomHelper(initialize_fn=initial_fn,
                                                    sample_fn=sample_fn,
                                                    next_inputs_fn=next_inputs_fn)

        def decode_with_attention(decoder_helper, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                # encoder_outputs:[encoder_sequence_length, batch, 2*hidden_size]
                # memory:[batch, encoder_sequence_length , 2*hidden_size]
                memory = tf.transpose(encoder_outputs, [1, 0, 2])
                # 加性attention, f(w1*h_i+w2*s_j)
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.hidden_size,
                    memory=memory,
                    memory_sequence_length=self.encoder_inputs_actual_length)
                cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=cell,
                    attention_mechanism=attention_mechanism,
                    attention_layer_size=self.hidden_size)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    cell=attn_cell,
                    output_size=self.slot_size,
                    activation=None,
                    reuse=reuse
                )

                # 定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
                # decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size,
                #                                                dtype=tf.float32).clone(cell_state=encoder_state)
                # 但此处我看并未用最后一个encoder的state赋值
                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell,
                    helper=decoder_helper,
                    initial_state=out_cell.zero_state(
                        dtype=tf.float32, batch_size=self.batch_size).clone(cell_state=self.encoder_final_state)
                    # initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size) # 原始作者写的是0,感觉有误
                    # initial_state=encoder_final_state)
                )
                # initial_state=encoder_final_state
                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, slot_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
                # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
                final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=True,
                    impute_finished=True,
                    maximum_iterations=self.input_steps
                )
                return final_outputs

        outputs = decode_with_attention(my_helper, 'decode')
        print("outputs: ", outputs)
        print("outputs.rnn_output: ", outputs.rnn_output) # [decoder_time_step, batch, slot_size]
        print("outputs.sample_id: ", outputs.sample_id) # [decoder_time_step, batch]
        # weights = tf.to_float(tf.not_equal(outputs[:, :-1], 0))
        self.decoder_prediction = outputs.sample_id
        # decoder_dim: slot_size
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(outputs.rnn_output))
        # decoder_targets_time_majored:[decoder_sequence_length, batch]
        self.decoder_targets_time_majored = tf.transpose(self.decoder_targets, [1, 0])
        # decoder_targets_true_length:[decoder_sequence_length, batch]
        self.decoder_targets_true_length = self.decoder_targets_time_majored[:decoder_max_steps]
        print("decoder_targets_true_length: ", self.decoder_targets_true_length)
        # 定义mask，使padding不计入loss计算
        self.mask = tf.to_float(tf.not_equal(self.decoder_targets_true_length, 0))
        # TODO:看了整个代码,感觉作者并没有将 last_slot_label信息加入 decoder的解码中
        # 定义slot标注的损失
        loss_slot = tf.contrib.seq2seq.sequence_loss(
            logits=outputs.rnn_output,
            targets=self.decoder_targets_true_length,
            weights=self.mask)
        # 定义intent分类的损失, intent_logits是rnn最后一层的hidden状态进行project
        # intent_logits:[batch, intent_size]
        # cross_entropy:[batch]
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.intent_targets, depth=self.intent_size, dtype=tf.float32),
            logits=intent_logits)
        loss_intent = tf.reduce_mean(cross_entropy)

        self.loss = loss_slot + loss_intent
        optimizer = tf.train.AdamOptimizer(name="a_optimizer")
        self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
        print("vars for loss function: ", self.vars)
        self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
        self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))
        # self.train_op = optimizer.minimize(self.loss)
        # train_op = layers.optimize_loss(
        #     loss, tf.train.get_global_step(),
        #     optimizer=optimizer,
        #     learning_rate=0.001,
        #     summaries=['loss', 'learning_rate'])

    def step(self, sess, mode, trarin_batch):
        """ perform each batch"""
        if mode not in ['train', 'test']:
            print(sys.stderr, 'mode is not supported!')
            sys.exit(1)
        unziped = list(zip(*trarin_batch))
        # print(np.shape(unziped[0]), np.shape(unziped[1]),
        #       np.shape(unziped[2]), np.shape(unziped[3]))
        if mode == 'train':
            output_feeds = [self.train_op, self.loss, self.decoder_prediction,
                            self.intent, self.mask, self.slot_W]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], axes=[1, 0]),
                         self.encoder_inputs_actual_length: unziped[1],
                         self.decoder_targets: unziped[2],
                         self.intent_targets: unziped[3]}
        if mode in ['test']:
            output_feeds = [self.decoder_prediction, self.intent]
            feed_dict = {self.encoder_inputs: np.transpose(unziped[0], [1, 0]),
                         self.encoder_inputs_actual_length: unziped[1]}

        results = sess.run(output_feeds, feed_dict=feed_dict)
        return results
