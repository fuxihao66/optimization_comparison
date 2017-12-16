import tensorflow as tf
from util import *
import timeit
import math

class Model:
	def __init__(self, num_features, num_classes):
		self.num_features = num_features
		self.num_classes = num_classes


		self.X = tf.placeholder("float", [None, num_features])
		self.Y = tf.placeholder("float",[None, num_classes])
		self.lr = tf.placeholder("float",[])
		self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
												initializer=tf.constant_initializer(0), trainable=False)
		self.build_forward()
		self.build_loss()
		self.build_accuracy()

		self.model_summary = tf.summary.merge_all()
	def build_forward(self):
		self.W = tf.get_variable('weight', shape=[self.num_features,self.num_classes], 
									dtype='float32', 
									initializer=tf.random_normal_initializer(mean=0.5))
		self.B = tf.get_variable('bias', shape=[self.num_classes], 
									dtype='float32', 
									initializer=tf.random_normal_initializer(mean=0.5))
		logit = tf.matmul(self.X, self.W) + self.B
		self.pY = tf.nn.softmax(logit, 1)

	def build_loss(self):
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pY, labels=self.Y))
		tf.add_to_collection('loss', self.loss)
		tf.summary.scalar(self.loss.op.name, self.loss)
	def build_accuracy(self):
		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pY,1), tf.argmax(self.Y,1)), "float"))
		tf.add_to_collection('accuracy', self.accuracy)
		tf.summary.scalar(self.accuracy.op.name, self.accuracy)


if __name__ == '__main__':		
	train_data, train_labels = read_data('train')
	train_data = train_data[:50000]
	train_labels = train_labels[:50000]
	num_features = train_data.shape[1]
	num_classes = train_labels.shape[1]
	
	batch_size = 100
	feed_lr = 0.001

	model = Model(num_features, num_classes)

	## Optimizer
	opt1 = tf.train.GradientDescentOptimizer(model.lr).minimize(model.loss, global_step=model.global_step)
	opt2 = tf.train.AdamOptimizer(model.lr).minimize(model.loss, global_step=model.global_step)
	opt3 = tf.train.AdadeltaOptimizer(model.lr).minimize(model.loss, global_step=model.global_step)

	# Create and initialize a session
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	init = tf.global_variables_initializer()
	sess.run(init)
	global_step = 0
	train_writer = tf.summary.FileWriter('/media/fuxihao/Data/MyDocuments/Optimization_2nd/tf_board', sess.graph)
	# training
	num_epochs = 2000
	
	for i in range(num_epochs):
		# start = timeit.default_timer()
		# if i % 50 == 0:
		# 	feed_lr = feed_lr*0.7
		avg_loss = 0.0
		batch_list = random_sampling(train_data, train_labels, batch_size)
		total_batch = len(batch_list)

		for batch in batch_list:

			global_step = sess.run(model.global_step) + 1
			summary, step_loss, operation = sess.run([model.model_summary, model.loss, opt2], feed_dict={model.X:batch['x'].todense(), model.Y:batch['y'].todense(), model.lr:feed_lr})
			avg_loss += step_loss
			train_writer.add_summary(summary, global_step)
		
		# stop = timeit.default_timer()
		print("epoch {} ended, avg loss is {}".format(i+1, avg_loss/total_batch))

	print(sess.run(model.B))


'''
	## start to eval the model with test data
	test_data, test_labels = read_data('test')
	test_data1 = test_data[:10000]
	# test_data2 = test_data[30000:]
	test_labels1 = test_labels[:10000]
	# test_labels2 = test_labels[30000:]
	accuracy_value1 = sess.run(model.accuracy, feed_dict={model.X:test_data1.todense(),model.Y:test_labels1.todense(), model.lr:feed_lr})
	# accuracy_value2 = sess.run(accuracy, feed_dict={X:test_data2.todense(), Y:test_labels2.todense(), lr:feed_lr})
	# accuracy_value = (accuracy_value1*test_data1.shape[0]+accuracy_value2*test_data2.shape[0])/test_data.shape[0]
	print('batch size is {}'.format(batch_size))
	# print('init lr is {}'.format(init_lr))
	print('accuracy value of test set is {}'.format(accuracy_value1))
'''