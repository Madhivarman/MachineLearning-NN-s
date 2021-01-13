"""
  Load Tensorflow ".pb" file to make predictions
"""

def load_file(parameters):

    model_file = parameters['model_file']
    image_to_predict = parameters['image_path']

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ['serve'], model_file)
        graph = tf.get_default_graph()
        
        #to see the AutoML graph
        # with open("model_network.txt", "a") as fp:
        #     for op in graph.get_operations():
        #         fp.write(op.name + "\t")
        #         fp.write(str(op.values()))
        #         fp.write("\n")
    
        print(" *** Session Created ***")
        
        #if you want to inspect input and output of the model graph
        #upload the .pb file to the NETRON app, and visualize the 
        #graph
        #predictions format
        #sess.run(['OUTPUT'], feed_dict={'INPUT_NAME':[img_file.read()]})
        with open(image_to_predict, "rb") as img_file:
            y_pred = sess.run(['Tile:0','scores:0'], feed_dict={'Placeholder:0':[img_file.read()]})
        
        predicted_class_scores = {} #dictionary

        labels, scores = y_pred[0][0].tolist(), y_pred[1][0].tolist()
        predicted_class_scores = {}

        for label, score in zip(labels, scores):
            predicted_class_scores[label] = score
    
    #filter by lower and upper bound
    lower = parameters['lower_range']
    higher = parameters['upper_range']

    possible_classes = []

    for k,v in predicted_class_scores.items():
        if v >= lower and v <= higher:
            possible_classes.append((k, v))
    
    return possible_classes
