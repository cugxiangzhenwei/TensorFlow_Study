import tensorflow as tf
a = tf.constant([1,2,3],name="a")
b = tf.constant([2,4,7],name="b")
result = a + b
sess = tf.Session()
print sess.run(result)
