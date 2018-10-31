import tensorflow as tf

class Model:
    # this here is for setting some hyperparameters
    def __init__(self, num_classes_arg=2, max_sequence_length=100, embedding_size=100, vocab_size=100, batch_size=5, l2_reg_lambda=0.0, learning_rate=1e-3, num_units_shared=32, num_units_aux=32, multitask=True, num_classes_aux=None, loss_weight_main=0.5, only_discourse=False):
        self.multitask = multitask
        self.only_discourse = only_discourse
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.input_x = tf.placeholder(tf.int32, [None, self.max_sequence_length], name="input_x")
        self.input_seqlen = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32, (), name="dropout_keep_prob")
        self.batch_size = batch_size
        self.l2_reg_lambda = l2_reg_lambda
        self.num_units_shared = num_units_shared
        self.learning_rate = learning_rate
        self.variable_memory = {}
        self.mask = None
        if only_discourse == False:
            self.num_classes_arg = num_classes_arg
            self.input_y_arg = tf.placeholder(tf.int64, [None, self.max_sequence_length, num_classes_arg],
                                              name="input_y_arg")
        if multitask == True or only_discourse == True:
            self.num_classes_aux = num_classes_aux
            #for citation auxiliary commented
            self.input_y_aux = tf.placeholder(tf.int64, [None, num_classes_aux], name="input_y_aux")
            #self.input_y_aux = tf.placeholder(tf.int64, [None, self.max_sequence_length, num_classes_aux], name="input_y_aux")
            self.num_units_aux = num_units_aux
            self.loss_weight_main = loss_weight_main


    def def_shared_bi_lstm_layer(self):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units_shared, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units_shared, state_is_tuple=True)

        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
        # None: it is initialized with zeros
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=self.embeddings,
            sequence_length=self.input_seqlen,
            initial_state_fw=None,
            initial_state_bw=None,
            dtype=tf.float32,
            parallel_iterations=None,
            swap_memory=False,
            time_major=False,
            scope=None
        )

        self.shared_fw_outputs, self.shared_bw_outputs = outputs
        self.shared_output_concat = tf.concat([self.shared_fw_outputs, self.shared_bw_outputs], axis=-1)
        fw_state, bw_state = states
        self.shared_state = tf.concat([fw_state, bw_state], 1)
        self.shared_state_size = self.num_units_shared


    def def_embedding_layer(self, use_pretrained_embeddings=False):
        with tf.name_scope("embedding_layer"):
            if not use_pretrained_embeddings:
                W = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
                self.embeddings = tf.nn.embedding_lookup(W, self.input_x)
            else:
                self.W_embeddings_init = tf.placeholder(tf.float32, shape=(self.vocab_size, self.embedding_size))
                W_embeddings = tf.Variable(self.W_embeddings_init, trainable=True, name="word_embeddings")
                self.embeddings = tf.nn.embedding_lookup(W_embeddings, self.input_x)


    def def_arg_predictions_loss_accuracy(self):
        with tf.name_scope("arg_prediction"):
            W = tf.get_variable("W_softmax_arg",
                shape=[2* self.shared_state_size, self.num_classes_arg],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes_arg]))

            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            output_flat = tf.reshape(self.shared_output_concat, [-1, 2 * self.shared_state_size])

            scores = tf.matmul(output_flat, W) + b

            self.arg_scores = tf.reshape(scores, [-1, self.max_sequence_length, self.num_classes_arg])
            self.arg_predictions = tf.argmax(self.arg_scores, 2)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.arg_scores, labels=self.input_y_arg)
            self.mask = tf.sequence_mask(self.input_seqlen, maxlen=self.max_sequence_length, dtype=tf.float32)
            losses *= self.mask
            mean_loss_by_example = tf.reduce_sum(losses, reduction_indices=1) / tf.to_float(self.input_seqlen)
            mean_loss = tf.reduce_mean(mean_loss_by_example)
            self.arg_loss = mean_loss + self.l2_reg_lambda * l2_loss

        with tf.name_scope("metrics"):
            self.arg_predictions_one_hot = tf.to_int64(tf.one_hot(self.arg_predictions, self.input_y_arg.get_shape().dims[2].value))



    def def_token_level_aux_predictions_loss_accuracy(self):
        with tf.name_scope("aux_prediction_token_level"):
            W = tf.get_variable("W_softmax_aux_token_level",
                shape=[2* self.shared_state_size, self.num_classes_aux],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes_aux]))
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            output_flat = tf.reshape(self.shared_output_concat, [-1, 2 * self.shared_state_size])

            scores = tf.matmul(output_flat, W) + b

            self.aux_scores = tf.reshape(scores, [-1, self.max_sequence_length, self.num_classes_aux])
            self.aux_predictions = tf.argmax(self.aux_scores, 2)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.aux_scores, labels=self.input_y_aux)

            self.mask = tf.sequence_mask(self.input_seqlen, maxlen=self.max_sequence_length, dtype=tf.float32)
            losses *= self.mask

            mean_loss_by_example = tf.reduce_sum(losses, reduction_indices=1) / tf.to_float(self.input_seqlen)
            mean_loss = tf.reduce_mean(mean_loss_by_example)

            self.aux_loss = mean_loss + self.l2_reg_lambda * l2_loss

        with tf.name_scope("metrics"):
            self.aux_predictions_one_hot = tf.to_int64(tf.one_hot(self.aux_predictions, self.input_y_aux.get_shape().dims[2].value))


    def def_attention(self):
        with tf.variable_scope('attention'):
            # the input to the attention has shape N, max_length, 2*state_size
            # the output of this should have shape N, 2* state_size
            # http://www.aclweb.org/anthology/N16-1174

            attention_vector = tf.get_variable("attention_vector", shape=[2 * self.shared_state_size],
                                               dtype=tf.float32)
            if self.mask is None:
                self.mask = tf.sequence_mask(self.input_seqlen, maxlen=self.max_sequence_length, dtype=tf.float32)
            attended_vector = tf.tensordot(self.shared_output_concat, attention_vector, axes=[[2], [0]])
            attended_vector *= self.mask

            attention_weights = self.softmax_ignore_zeros(attended_vector, none_dim_replacement=self.batch_size)
            attention_weights = tf.expand_dims(attention_weights, -1)
            self.attention_output = tf.matmul(self.shared_output_concat, attention_weights, transpose_a=True)
            self.attention_output = tf.squeeze(self.attention_output, axis=2)



    def def_aux_bi_lstm_layer(self):
        with tf.variable_scope("aux_bi_lstm_layer"):
            fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units_aux, state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units_aux, state_is_tuple=True)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)

            self.attention_output = tf.expand_dims(self.attention_output, axis=1)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=self.attention_output,
                initial_state_fw=None,
                initial_state_bw=None,
                dtype=tf.float32,
                parallel_iterations=None,
                swap_memory=False,
                time_major=False,
                scope=None
            )

            self.aux_fw_outputs, self.aux_bw_outputs = outputs
            self.aux_output_concat = tf.concat([self.aux_fw_outputs, self.aux_bw_outputs], axis=-1)
            fw_state, bw_state = states
            self.aux_state = tf.concat([fw_state, bw_state], 1)
            self.aux_state_size = self.num_units_aux


    def def_aux_predictions_loss_accuracy(self):
        with tf.name_scope("auxiliary_prediction"):
            W = tf.get_variable("W_softmax_aux",
                shape=[2* self.aux_state_size, self.num_classes_aux],
                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes_aux]))

            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            output_flat = tf.reshape(self.aux_output_concat, [-1, 2 * self.aux_state_size])

            scores = tf.matmul(output_flat, W) + b

            self.aux_scores = tf.reshape(scores, [-1, self.num_classes_aux])
            self.aux_predictions = tf.argmax(self.aux_scores, 1)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.aux_scores, labels=self.input_y_aux)
            self.aux_loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss


        with tf.name_scope("metrics"):
            self.aux_predictions_one_hot = tf.to_int64(
                tf.one_hot(self.aux_predictions, self.input_y_aux.get_shape().dims[1].value))


    def def_multitask_uncertainty_loss(self):
        self.sigma_1 = tf.get_variable("sigma_1", dtype=tf.float32, initializer=tf.constant(0.5))
        self.sigma_2 = tf.get_variable("sigma_2", dtype=tf.float32, initializer=tf.constant(0.5))
        self.sigma_1 = tf.Print(self.sigma_1, [self.sigma_1], message="sigma 1")
        self.sigma_2 = tf.Print(self.sigma_2, [self.sigma_2], message="sigma 2")
        self.loss = tf.multiply(tf.divide(1.0, tf.multiply(2.0, tf.square(self.sigma_1))), self.arg_loss) \
                    + tf.log(tf.square(self.sigma_1)) \
                    + tf.multiply(tf.divide(1.0, tf.multiply(2.0, tf.square(self.sigma_2))), self.aux_loss) \
                    + tf.log(tf.square(self.sigma_2))

    def def_total_loss(self):
        if self.multitask == True:
            self.loss = 0.5 * self.arg_loss + 0.5 * self.aux_loss
        elif self.only_discourse == True:
            self.loss = self.aux_loss
        else:
            self.loss = self.arg_loss



    def softmax_ignore_zeros(self, tensor, none_dim_replacement=None):
        nonzero_softmaxes = tf.where(tf.not_equal(tensor, tf.constant(0, dtype=tf.float32)), tf.exp(tensor), tensor)
        reshape_dims = tensor.get_shape().as_list()
        if reshape_dims[0] is None and none_dim_replacement is not None:
            reshape_dims[0] = none_dim_replacement
        reshape_dims[-1] = 1

        norms = tf.reshape(tf.reduce_sum(nonzero_softmaxes, axis=1), shape=reshape_dims)
        return tf.div(nonzero_softmaxes, norms)



    def def_joint_train_operation(self, global_step=0):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(loss=self.loss, global_step=global_step)


    def def_alternate_train_operation(self, global_step=0):
        self.train_step_arg = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.arg_loss, global_step=global_step)
        self.train_step_aux = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.aux_loss,
                                                                                  global_step=global_step)

    def get_hyperparameters(self):
        if self.multitask == True:
            params = {"embedding_size": self.embedding_size, "l2_reg_lambda": self.l2_reg_lambda, "dropout_keep_prob": self.dropout_keep_prob,
                      "num_units_shared": self.num_units_shared, "num_units_aux": self.num_units_aux, "learning_rate": self.learning_rate}
        elif self.only_discourse == True:
            params = {"embedding_size": self.embedding_size, "l2_reg_lambda": self.l2_reg_lambda, "dropout_keep_prob": self.dropout_keep_prob,
                      "num_units_shared": self.num_units_shared, "num_units_aux": self.num_units_aux, "learning_rate": self.learning_rate}
        else:
            params = {"embedding_size": self.embedding_size, "l2_reg_lambda": self.l2_reg_lambda, "dropout_keep_prob": self.dropout_keep_prob,
                      "num_units_shared": self.num_units_shared, "learning_rate": self.learning_rate}
        return params


    def get_variable_values(self, session):
        variables = {}
        for v in self.variable_memory:
            value = self.variable_memory[v].eval(session=session)
            variables[v] = value
        return variables


    def get_model(self, session):
        return [self.get_hyperparameters(), self.get_variable_values(session)]