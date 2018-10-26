import tensorflow as tf

def compute1():
	a = tf.constant(5)
	b = tf.constant(3)

	d = a + b
	c = a * b 
	f = d + c
	e = c - d
	g = f / e
	sess = tf.Session()
	out = sess.run(g)
	sess.close()
	print("out1 equal {}".format(out))


def compute2():
	a = tf.constant(5, dtype= tf.float32)
	b = tf.constant(3, dtype = tf.float32)

	c = a * b 
	d = tf.sin(c)
	e = b / d 
	sess = tf.Session()
	out = sess.run(e)

	sess.close()
	print("out2 equal {}".format(out))

compute1()
compute2()







