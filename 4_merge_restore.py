import tensorflow as tf

model_path = 'merge_model.pb'


sess = tf.Session()
with tf.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


tf.import_graph_def(graph_def, name='XXX')
graph = tf.get_default_graph()

input_x  = graph.get_tensor_by_name('XXX/x:0')
input_y = graph.get_tensor_by_name('XXX/y:0')
output_op = graph.get_tensor_by_name('XXX/mod_0:0')
output_op_1 = graph.get_tensor_by_name('XXX/mod_1:0')
output_op_2 = graph.get_tensor_by_name('XXX/mod_2:0')
output_op_3 = graph.get_tensor_by_name('XXX/mod_3:0')
output_op_4 = graph.get_tensor_by_name('XXX/mod_4:0')

print(input_x)
print(input_y)
print(output_op)
print(output_op_1)

ret = sess.run([output_op,output_op_1,output_op_2,output_op_3,output_op_4],
            feed_dict={input_x: 5, input_y: 5})
print(ret)

builder = tf.saved_model.builder.SavedModelBuilder("./success/1/" )
inputs = {
    "input_x":tf.saved_model.utils.build_tensor_info(input_x),
    "input_y":tf.saved_model.utils.build_tensor_info(input_y)
}
outputs = {
    "mod_0":tf.saved_model.utils.build_tensor_info(output_op),
    "mod_1":tf.saved_model.utils.build_tensor_info(output_op_1),
    "mod_2":tf.saved_model.utils.build_tensor_info(output_op_2),
    "mod_3":tf.saved_model.utils.build_tensor_info(output_op_3),
    "mod_4":tf.saved_model.utils.build_tensor_info(output_op_4)
}
prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
    )
    )
builder.add_meta_graph_and_variables(
    sess,
    [tf.saved_model.tag_constants.SERVING],
    signature_def_map={"predictions": prediction_signature},
    main_op=tf.tables_initializer(),
    strip_default_attrs=True
    )
builder.save()

