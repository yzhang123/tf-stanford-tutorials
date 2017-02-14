
import tensorflow as tf 

a = tf.Variable(2)
b = tf.Variable(3)
c = tf.add(a, b)
W = tf.Variable(3)
op = W.assign(5)
init = tf.global_variables_initializer()
#d = tf.ones_like(t_1)init = 
with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	sess.run(init)
	print W.eval()
	print sess.run(op)
writer.close()


