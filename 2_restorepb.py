import tensorflow as tf

model_path = 'model.pb'


sess = tf.Session()
with tf.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


tf.import_graph_def(graph_def, name='XXX')
graph = tf.get_default_graph()


input_x  = graph.get_tensor_by_name('XXX/x:0')
input_y = graph.get_tensor_by_name('XXX/y:0')
output_op = graph.get_tensor_by_name('XXX/op_to_store:0')
output_op_1 = graph.get_tensor_by_name('XXX/op_1_to_store:0')
bbbb = graph.get_tensor_by_name('XXX/b:0')

print(input_x)
print(input_y)
print(output_op)
print(output_op_1)


#sess.run(tf.global_variables_initializer())

print(sess.run('XXX/b:0'))

ret = sess.run([output_op,output_op_1],  feed_dict={input_x: 5, input_y: 5})
print(ret)
