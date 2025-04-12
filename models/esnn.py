import tensorflow as tf
from dataset.dataset_to_sklearn import *
from dataset.makeTrainingData import makeSmartNData
from models.eval import eval_dual_ann
from models.utils import normalizeBatchSize
from utils.keras_utils import keras_sqrt_diff
from utils.KerasCallbacks import CustomModelCheckPoint, callbackdict
from utils.plotting_utils import plot_training_progress


def esnn(
    o_X,
    o_Y,
    X,
    Y,
    datasetname,
    regression=False,
    epochs=2000,
    val_split=0,
    shuffle=True,
    batch_size=32,
    optimizer=None,
    onehot=True,
    multigpu=False,
    callbacks=None,
    trainratio=0.2,
    networklayers=[13, 13],
    rootdir="rootdir",
    alpha=0.8,
    makeTrainingData=None,
    n=0,
):

    if makeTrainingData == None:
        makeTrainingData = makeSmartNData

    model, embeddingmodel = make_eSNN_model(X, Y, networklayers, regression)

    batch_size = normalizeBatchSize(X, batch_size)

    if regression != True:
        loss_dict = {  #'dist_output': 'mean_squared_error',
            "dist_output": "binary_crossentropy",
            "class1_output": "categorical_crossentropy",
            "class2_output": "categorical_crossentropy",
        }
        lossweight_dict = {
            "dist_output": alpha,
            "class1_output": (1.0 - alpha) / 2.0,
            "class2_output": (1.0 - alpha) / 2.0,
        }
    else:
        loss_dict = {
            "dist_output": "mean_squared_error",
            "reg_output1": "mean_squared_error",
            "reg_output2": "mean_squared_error",
        }
        lossweight_dict = {
            "dist_output": alpha,
            "reg_output1": (1.0 - alpha) / 2.0,
            "reg_output2": (1.0 - alpha) / 2.0,
        }

    model.compile(
        optimizer=optimizer["constructor"](),
        loss=loss_dict,
        metrics={
            "dist_output": ["accuracy"],
            "class1_output": ["accuracy"],
            "class2_output": ["accuracy"],
        },
        loss_weights=lossweight_dict,
    )

    features, targets, Y1, Y2 = makeTrainingData(X, Y, regression, distance=True)
    training_data = [
        features[:, 0 : X.shape[1]],
        features[:, X.shape[1] : 2 * X.shape[1]],
    ]

    target_data = [targets, Y1, Y2]

    run_callbacks = list()
    ret_callbacks = dict()
    filepath = rootdir + "esnn-weights.best.hdf5"
    for callback in callbacks:
        cbo = callbackdict[callback]["callback"](
            o_X,
            o_Y,
            X,
            Y,
            batch_size,
            eval_dual_ann,
            datasetname,
            filepath,
            save_best_only=True,
        )
        run_callbacks.append(cbo)
        ret_callbacks[callback] = cbo
    filepath = rootdir + "saved-model-{epoch:02d}-{accuracy:.2f}.hdf5"
    run_callbacks.append(CustomModelCheckPoint(filepath="esnn", rootdir=rootdir, n=n))

    # test = np.hstack((features, targets))
    batch_size = features.shape[0]
    history = model.fit(
        training_data,
        target_data,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=run_callbacks,
    )

    figure_file = plot_training_progress(history, datasetname, rootdir)
    print(f"Training progress saved to {figure_file}")

    return model, history, ret_callbacks, embeddingmodel


def make_eSNN_model(X, Y, networklayers, regression=False):
    g_layers = networklayers
    c_layers = networklayers

    if isinstance(networklayers[0], list):
        g_layers = networklayers[0]
        c_layers = networklayers[1]

    input1 = tf.keras.Input(shape=(X.shape[1],), dtype="float32")
    input2 = tf.keras.Input(shape=(X.shape[1],), dtype="float32")

    # make G(x)
    t1 = input1
    t2 = input2
    for networklayer in g_layers:
        dl1 = tf.keras.layers.Dense(
            int(networklayer), activation="relu"
        )  # ,activity_regularizer=l2(0.01))
        t1 = dl1(t1)
        t2 = dl1(t2)

    dl1.name = "embeddinglayer"

    # subtracted = Subtract()([encoded_i1,encoded_i2])

    # TODO: We had 5 layers here (from subtract to output), maybe compare
    # different "top-layers" vs (the combination) "bottom-layers", in which
    # case we need more than one layers parameter

    # make C(x,y)
    o_t = tf.keras.layers.Lambda(keras_sqrt_diff)([t1, t2])
    for networklayer in c_layers:
        o_t = tf.keras.layers.Dense(int(networklayer), activation="relu")(o_t)

    # Similarity output
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="dist_output")(o_t)

    # Make class output from G(x) to get two more signal sources
    if regression == True:  # Regression or 1 output classification
        inner_output1 = tf.keras.layers.Dense(
            Y.shape[1],
            activation="linear",
            kernel_initializer="random_uniform",
            name="reg_output1",
        )
        inner_output2 = tf.keras.layers.Dense(
            Y.shape[1],
            activation="linear",
            kernel_initializer="random_uniform",
            name="reg_output2",
        )
    else:  # onehot
        inner_output1 = tf.keras.layers.Dense(
            Y.shape[1], activation="softmax", name="class1_output"
        )
        inner_output2 = tf.keras.layers.Dense(
            Y.shape[1], activation="softmax", name="class2_output"
        )

    # Classification task output
    output1 = inner_output1(t1)
    output2 = inner_output2(t2)

    model = tf.keras.Model(inputs=[input1, input2], outputs=[output, output1, output2])

    embeddingmodel = tf.keras.Model(inputs=[input1], outputs=[t1])
    return model, embeddingmodel
