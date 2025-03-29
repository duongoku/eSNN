import tensorflow as tf

from models.rprop import RProp

__name__ = "dataset_to_models"


def makeNormalModelLayers(
    n, inp, networklayers, regression, activation_function, layernameprefix=""
):
    # making the the first layer
    layers = list()
    input = tf.keras.Input(
        shape=(inp,), dtype="float32", name=layernameprefix + "input"
    )
    last_layer = tf.keras.layers.Dense(
        activation=activation_function,
        units=int(networklayers[0]),
        kernel_initializer="random_uniform",
        name=layernameprefix + "secondlayer",
    )(input)
    layers.append(last_layer)
    # then creating the hidden layers based on the lists received in
    # networklayers
    for i in range(0, len(networklayers) - 1):
        this_layer = tf.keras.layers.Dense(
            int(networklayers[i]),
            kernel_initializer="random_uniform",
            activation=activation_function,
            name=layernameprefix + "layer" + str(i),
        )(last_layer)
        last_layer = this_layer
        layers.append(last_layer)

    # making the last layer, this has to be different depending on if this
    # network is doing regression or classification
    output = None
    if regression is True:  # regression or 1 output classification
        output = tf.keras.layers.Dense(
            n,
            activation="linear",
            kernel_initializer="random_uniform",
            name=layernameprefix + "output",
        )(last_layer)
    else:  # onehot
        output = tf.keras.layers.Dense(
            n,
            activation="softmax",
            kernel_initializer="random_uniform",
            name=layernameprefix + "output",
        )(last_layer)
    layers.append(output)
    return input, output, layers


def makeAndCompileNormalModel(
    n,
    inp,
    networklayers,
    optimizer,
    regression,
    onehot,
    multigpu,
    activation_function,
    layernameprefix="",
):
    input, output, layers = makeNormalModelLayers(
        n=n,
        inp=inp,
        networklayers=networklayers,
        regression=regression,
        activation_function=activation_function,
        layernameprefix=layernameprefix,
    )

    # Create distribution strategy if multigpu is enabled
    if multigpu:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = tf.keras.Model(inputs=input, outputs=output)
            if regression is True:
                model.compile(
                    loss="mean_squared_error", optimizer="adam", metrics=["accuracy"]
                )
            else:
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"],
                )
    else:
        model = tf.keras.Model(inputs=input, outputs=output)
        if regression is True:
            model.compile(
                loss="mean_squared_error", optimizer="adam", metrics=["accuracy"]
            )
        else:
            model.compile(
                loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
            )

    return model, output


def makeGabelClassifierModel(inp, networklayers):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(inp,)))
    model.add(
        tf.keras.layers.Dense(
            inp,
            activation="sigmoid",
            kernel_initializer="random_uniform",
        )
    )

    for layer in networklayers:
        model.add(
            tf.keras.layers.Dense(
                int(layer), kernel_initializer="random_uniform", activation="sigmoid"
            )
        )
    model.add(
        tf.keras.layers.Dense(
            1, activation="sigmoid", kernel_initializer="random_uniform"
        )
    )
    model.compile(loss="mean_squared_error", optimizer=RProp(), metrics=["accuracy"])
    return model


def savemodel(model, modelfile):
    # serialize model to JSON
    model_json = model.to_json()
    with open(modelfile + ".json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(modelfile + ".weights.h5")
        print(
            "saved model object to %s and weights to %s "
            % (modelfile + ".json", modelfile + ".weights.h5")
        )


def makeANNModel(
    o_X,
    o_Y,
    X,
    Y,
    regression=False,
    epochs=2000,
    val_split=0,
    shuffle=True,
    batch_size=32,
    optimizer=None,
    onehot=True,
    multigpu=False,
    callbacks=None,
    networklayers=[13, 13],
    rootdir="rootdir",
    alpha=0.8,
    makeTrainingData=None,
):
    # print(f"shape is larger than one, batch size is {batch_size}")
    model = makeAndCompileNormalModel(
        Y.shape[1], X.shape[1], networklayers, optimizer, regression, onehot, multigpu
    )

    return model


# t4i4 makes one G(x), trains it on dataset, copies it, sharing weights.
# adds trainable C(x,y), trains whole stack


def create_base_network(input_shape, networklayers=[13, 13]):
    """Base network to be shared (eq. to feature extraction)."""
    input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(int(networklayers[0]), activation="relu")(input)
    for i in range(0, len(networklayers) - 1):
        x = tf.keras.layers.Dense(int(networklayers[i]), activation="relu")(x)
    return tf.keras.Model(input, x)


# t4i4 makes one G(x), trains it on dataset, copies it, sharing weights.
# adds trainable C(x,y), trains whole stack


def loss(y_true, y_pred, alpha):
    y_true_f = tf.keras.ops.reshape(y_true, (-1,))
    y_pred_f = tf.keras.ops.reshape(y_pred, (-1,))
    intersection = tf.keras.ops.sum(y_true_f * y_pred_f)
    return tf.keras.ops.abs(y_true - y_pred)


def my_loss(
    alpha,
):
    def dice(y_true, y_pred):
        return -loss(y_true, y_pred, alpha)

    return dice


def addToNetStack(layerlist, input):
    layer = layerlist.pop()
    if len(layerlist) == 0:
        return layer(input)
    else:
        return layer(addToNetStack(layerlist, input))


def create_shared_weights(conv1, conv2, input_shape):
    with tf.keras.name_scope(conv1.name):
        conv1.build(input_shape)
    with tf.keras.name_scope(conv2.name):
        conv2.build(input_shape)
    conv2.kernel = conv1.kernel
    conv2.bias = conv1.bias
    conv2._trainable_weights = []
    conv2._trainable_weights.append(conv2.kernel)
    conv2._trainable_weights.append(conv2.bias)
