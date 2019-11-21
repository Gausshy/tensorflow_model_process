import argparse
import tensorflow as tf


def load_graph(frozen_graph_filename,name):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
                    graph_def, 
                    input_map=None, 
                    return_elements=None, 
                    name=name, 
                    op_dict=None, 
                    producer_op_list=None
        )
    return graph


def load_graph_def(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def combine():
    model_path = "model.pb"
    out_pb_path = "./"
    with tf.Graph().as_default() as g_combined:
        with tf.Session(graph=g_combined) as sess:
            graph_def_0 = load_graph_def(model_path)
            graph_def_1 = load_graph_def(model_path)
            graph_def_2 = load_graph_def(model_path)
            graph_def_3 = load_graph_def(model_path)
            graph_def_4 = load_graph_def(model_path)
            
            # 定义新输入
            input_x = tf.placeholder(tf.int32, name='x') 
            input_y = tf.placeholder(tf.int32, name='y')

            # 重定向输入输出
            #graph_0 = tf.import_graph_def(graph_def_0, input_map={"model_0/x:0":input_x, "model_0/y:0":input_y}, return_elements=["model_0/op_to_store:0","model_0/op_1_to_store:0"]) 
            graph_0 = tf.import_graph_def(graph_def_0, input_map={"x:0":input_x, "y:0":input_y}, return_elements=["op_to_store:0","op_1_to_store:0"]) 
            # 定义新的输出节点
            tf.identity(graph_0, "mod_0")

            # 重定向输入输出
            #graph_1 = tf.import_graph_def(graph_def_1, input_map={"model_1/x:0":input_x, "model_1/y:0":input_y}, return_elements=["model_1/op_to_store:0","model_1/op_1_to_store:0"]) 
            graph_1 = tf.import_graph_def(graph_def_1, input_map={"x:0":input_x, "y:0":input_y}, return_elements=["op_to_store:0","op_1_to_store:0"]) 
            # 定义新的输出节点
            tf.identity(graph_1, "mod_1")
            
            # 重定向输入输出
            #graph_0 = tf.import_graph_def(graph_def_0, input_map={"model_0/x:0":input_x, "model_0/y:0":input_y}, return_elements=["model_0/op_to_store:0","model_0/op_1_to_store:0"]) 
            graph_2 = tf.import_graph_def(graph_def_2, input_map={"x:0":input_x, "y:0":input_y}, return_elements=["op_to_store:0","op_1_to_store:0"]) 
            # 定义新的输出节点
            tf.identity(graph_2, "mod_2")

            # 重定向输入输出
            #graph_0 = tf.import_graph_def(graph_def_0, input_map={"model_0/x:0":input_x, "model_0/y:0":input_y}, return_elements=["model_0/op_to_store:0","model_0/op_1_to_store:0"]) 
            graph_3 = tf.import_graph_def(graph_def_3, input_map={"x:0":input_x, "y:0":input_y}, return_elements=["op_to_store:0","op_1_to_store:0"]) 
            # 定义新的输出节点
            tf.identity(graph_3, "mod_3")

            # 重定向输入输出
            #graph_0 = tf.import_graph_def(graph_def_0, input_map={"model_0/x:0":input_x, "model_0/y:0":input_y}, return_elements=["model_0/op_to_store:0","model_0/op_1_to_store:0"]) 
            graph_4 = tf.import_graph_def(graph_def_4, input_map={"x:0":input_x, "y:0":input_y}, return_elements=["op_to_store:0","op_1_to_store:0"]) 
            # 定义新的输出节点
            tf.identity(graph_4, "mod_4")

            # 模型结合
            g_combined_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,["mod_0","mod_1","mod_2","mod_3","mod_4"])
            # 大图生成新的pb
            tf.train.write_graph(g_combined_def, out_pb_path, 'merge_model.pb', as_text=False)


if __name__ == '__main__':
    combine()
    '''
    frozen_model_filename = "results/frozen_model.pb"
    graph = load_graph(frozen_model_filename)
    for op in graph.get_operations():
        print(op.name,op.values())

    x = graph.get_tensor_by_name('prefix/placeholder/inputs_placeholder:0')
    y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
        })
        print(y_out)
    print ("finish")
    '''
