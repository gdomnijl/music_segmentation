import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
PRINT_LOSS_EVERY = 5
## cnn_config: 
# 

## bilstm_config:

class CNN_biLSTM_Model:
	def __init__(self, cnn_config, bilstm_config, train_config):
		#self.inputs = inputs
		self.cnn_filter_config = cnn_config['filter_config']
		self.cnn_regularization = cnn_config['regularization']
		self.cnn_dropout = cnn_config['cnn_dropout']
		self.fc_regularization = cnn_config['fc_regularization']

		self.lstm_input_dim = bilstm_config['input_dim']
		self.lstm_output_dims = bilstm_config['output_dims'] ## a list encoding info on number of stacked layer as well 
		self.sequence_len = bilstm_config['sequence_len']
		self.lstm_dropout = bilstm_config['lstm_dropout']

		self.grad_clipping = train_config['grad_clipping']

	def _add_placeholders(self):
		## TODO: make the last dimension dynamic while fc dense layer don't complain
		self.inputs = tf.placeholder(shape = [None, 1, 128, 128], dtype = tf.float32)
		self.labels = tf.placeholder(shape = [None, self.sequence_len], dtype = tf.int32)
		self.learning_rate = tf.placeholder(shape = None, dtype = tf.float32) ##TODO: weird to have it as a placeholder but ...
		#self.batch_size = tf.placeholder
	def _add_logits(self):
		with tf.variable_scope('cnn'):
			#N,T,C,H,W = self.inputs.get_shape().as_list()
			input_layer = tf.transpose(self.inputs, [0,2,3,1])#tf.reshape(self.inputs, [*(N,T), C, H ,W])
			num_layer = len(self.cnn_filter_config['filter_size'])
			filter_config = self.cnn_filter_config
			for i in range(num_layer):
				
				filter_size = filter_config['filter_size'][i]
				num_filter = filter_config['num_filter'][i]
				pool_size = filter_config['pool_size'][i]
				pool_stride = filter_config['pool_stride'][i]

				conv_relu = tf.layers.conv2d(
					inputs = input_layer,
					filters = num_filter,
					kernel_size = filter_size, #[4,4],
					padding = "same",
					activation = tf.nn.relu,
					#data_format = 'channels_first',
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.cnn_regularization))

				pool = tf.layers.max_pooling2d(
					inputs = conv_relu, 
					pool_size = pool_size, 
					strides = pool_stride)
					#data_format = 'channels_first')

				##TODO: add batchnorm here
				
				input_layer = tf.nn.dropout(pool, self.cnn_dropout)
			
			cnn_output = tf.transpose(input_layer, [0,3,1,2])
			#tf.layers.Flatten()(input_layer)

		with tf.variable_scope('fc'):
			fc_output = tf.layers.dense(
				inputs = cnn_output, units = self.lstm_input_dim,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.fc_regularization)) ## fc
			print("#####",tf.shape(fc_output))

		with tf.variable_scope('bi_lstm'):
			## TODO: reg for lstm
			cell_fw = tf.nn.rnn_cell.MultiRNNCell(
				[tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_output_dims]
				)
			cell_bw = tf.nn.rnn_cell.MultiRNNCell(
				[tf.contrib.rnn.BasicLSTMCell(size) for size in self.lstm_output_dims]
				)
			
			## check: tf dynamically get shape
			C = fc_output.get_shape().as_list()[1]
			H = fc_output.get_shape().as_list()[2]
			W = fc_output.get_shape().as_list()[3]
			T = self.sequence_len
			lstm_input = tf.reshape(fc_output, [-1, T, C*H*W])

			
			(output_fw, output_bw), _  = tf.nn.bidirectional_dynamic_rnn(
				cell_fw,cell_bw, 
				lstm_input,
				dtype = tf.float32
				#sequence_length = [self.sequence_len]*self.batch_size
				)
			
			##TODO: add batch norm here
			
			lstm_output = tf.concat([output_fw, output_bw], axis = -1)
			lstm_output = tf.nn.dropout(lstm_output, self.lstm_dropout)


		with tf.variable_scope('logit'):
			self.logits = tf.nn.softmax(lstm_output, axis = -1)
			self.pred = tf.argmax(self.logits, axis = -1)


	def _add_loss(self):

		self.loss = 0
		
		losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits, labels = self.labels)
		## TODO: using mean after masking?
		self.loss = tf.reduce_mean(losses)
		print("##### loss after reduce mean:", self.loss)

	def _add_train(self):

		with tf.variable_scope('train_step'):
			tvars = tf.trainable_variables()
			
			if self.grad_clipping:
				grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 100.0) ##TODO: is 100 the right choice ?
			else:
				grad = tf.gradients(self.loss, tvars)
			## TODO: make train options
			optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)#=1e-3)
			self.train_op = optimizer.apply_gradients(zip(grad, tvars))
    ## TODO: logging
	def build(self):
		self._add_placeholders()
		self._add_logits()
		self._add_loss()
		self._add_train()

    
	def build_batch_for_epoch(self, data, labels, batch_index, batch_size):
    	# note: data is in (N, 7, 1, 128, 128)
    	# note: labels is in (N x 7 x num_classes)
		X_batch = data[batch_index:batch_index+batch_size]
		y_batch = labels[batch_index:batch_index+batch_size]
		return X_batch, y_batch


	def train(self, X_train, y_train, epochs, batch_size, model_name, learning_rate = 1e-3): ## early stoping
    	## TODO: model saving
		tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			N, T, C, H, W = X_train.shape 
			num_batch = int(N / batch_size)
			print("num_batch ", num_batch)
			#X_train_seq = X_train.reshape([-1, self.sequence_len, C, H, W])
			for e in range(epochs):
				X_shuffled, y_shuffled = shuffle(
    				X_train,
    				y_train) ## will only shuffle the first dimension
				
				for i in range(num_batch):
					## if last batch of the epoch
					if i == num_batch - 1:
						leftover = N % batch_size - 1
						X_batch, y_batch = self.build_batch_for_epoch(X_shuffled, y_shuffled, i, leftover)
					else:
						X_batch, y_batch = self.build_batch_for_epoch(X_shuffled, y_shuffled, i, batch_size)
					X_batch = X_batch.reshape(-1, C, H, W)
					feed_dict = {
						self.inputs: X_batch,
						self.labels: y_batch,
						self.learning_rate: learning_rate
					}

					_, loss_train =sess.run([self.train_op,self.loss], feed_dict = feed_dict)

					if i % PRINT_LOSS_EVERY == 0:
						print("Training loss at epoch %d batch %d: %f" % (e, i, loss_train))

			saver.save(sess, model_name)
				## TODO: if early stopping
				## TODO: model saving

	def predict(self, X_test, y_test, model_name):
		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess, model_name)

			pred = sess.run(self.labels, feed_dict = {
					self.inputs: X_test,
					self.labels: y_test
				})

		# for i in y_test.shape[0]:
		# 	y_seq = y_test[i, :]
		# 	pred_seq = pred[i, :]
			
		return pred

	def e




cnn_config = {'filter_config': {'filter_size': [4, 4],
								'num_filter': [5, 10],
								'pool_size': [2, 2],
								'pool_stride': [2, 2]
								},
			  'regularization': 0.01,
			  'cnn_dropout': 1.0,
			  'fc_regularization': 0.01}
			  

bilstm_config = {'input_dim': 64,
			  'output_dims': [128, 64, 6],## 6 being the num_of_classes TODO: what could be a better choice of size 
			   'sequence_len': 7,
			   'lstm_dropout': 1.0} 

train_config = {'grad_clipping': False}
model = CNN_biLSTM_Model(cnn_config, bilstm_config, train_config)
model.build()

N = 50
T = 7
C = 1
H = 128
W = 128
X_train = np.random.rand(N, T, C, H, W)
y_train = np.random.randint(0, 6, size = (N, T))

batch_size = 8
epoch = 1


model.train(X_train, y_train, epoch, batch_size)
