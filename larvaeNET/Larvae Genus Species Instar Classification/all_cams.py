import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

def grad_cam_mine(grad_model, input, height, width, threshold = None):
    last_conv_layer_output = grad_model(np.array([input]).astype('float32'))
    pooled_grads = tf.reduce_mean(last_conv_layer_output, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = np.array(heatmap)
    heatmap = np.uint8(255 * heatmap)
    
    if threshold is not None: heatmap = heatmap * (heatmap > threshold)
    
    heatmap = cv2.applyColorMap(cv2.resize(heatmap,(width, height)), cv2.COLORMAP_JET)

    return heatmap

def cam(model, grad_model, input, model_name, height, width, threshold = None):
    last_conv_layer_output = grad_model(np.array([input]).astype('float32'))
    last_conv_layer_output = last_conv_layer_output[0]

    # last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, layers.Conv2D))
    # target_layer = model.get_layer(last_conv_layer.name)
    # class_weights = target_layer.get_weights()[0]
    
    if model_name == "efficientNet": num = 472
    else: num = 777
    
    class_weights = model.layers[num].get_weights()[0]
    
    heatmap = tf.reduce_mean(class_weights * last_conv_layer_output, axis=(2))

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = np.array(heatmap)
    heatmap = np.uint8(255 * heatmap)
    
    if threshold is not None: heatmap = heatmap * (heatmap > threshold)
    
    heatmap = cv2.applyColorMap(cv2.resize(heatmap,(width, height)), cv2.COLORMAP_JET)

    return heatmap


def grad_cam_keras(model, image, height, width, threshold = None, interpolant=0.5):

    """VizGradCAM - Displays GradCAM based on Keras / TensorFlow models
    using the gradients from the last convolutional layer. This function
    should work with all Keras Application listed here:
    https://keras.io/api/applications/
    Parameters:
    model (keras.model): Compiled Model with Weights Loaded
    image: Image to Perform Inference On
    plot_results (boolean): True - Function Plots using PLT
                            False - Returns Heatmap Array
    Returns:
    Heatmap Array?
    """
    #sanity check
    assert (interpolant > 0 and interpolant < 1), "Heatmap Interpolation Must Be Between 0 - 1"
    #STEP 1: Preprocesss image and make prediction using our model
    #input image
    original_img = np.asarray(image, dtype = np.float32)
    #expamd dimension and get batch size
    img = np.expand_dims(original_img, axis=0)
    #predict
    prediction = model.predict(img)
    #prediction index
    prediction_idx = np.argmax(prediction)
    #STEP 2: Create new model
    #specify last convolutional layer
    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, layers.Conv2D))
    target_layer = model.get_layer(last_conv_layer.name)
    #compute gradient of top predicted class
    with tf.GradientTape() as tape:
        #create a model with original model inputs and the last conv_layer as the output
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        #pass the image through the base model and get the feature map  
        conv2d_out, prediction = gradient_model(img)
        #prediction loss
        loss = prediction[:, prediction_idx]
    #gradient() computes the gradient using operations recorded in context of this tape
    gradients = tape.gradient(loss, conv2d_out)
    #obtain the output from shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
    output = conv2d_out[0]
    #obtain depthwise mean
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))
    #create a 7x7 map for aggregation
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    #multiply weight for every layer
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]
    activation_map = activation_map.numpy()
    
    #ensure no negative number
    activation_map = np.maximum(activation_map, 0)
    #convert class activation map to 0 - 255
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    #rescale and convert the type to int
    activation_map = np.uint8(255 * activation_map)
    #Threshold apply
    if threshold is not None: activation_map = activation_map * (activation_map > threshold)
    #resize to image size
    activation_map = cv2.resize(activation_map, 
                                (width, height))
    #convert to heatmap
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    #superimpose heatmap onto image
    # original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
    # cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # cvt_heatmap = img_to_array(cvt_heatmap)
    #enlarge plot
    return heatmap

