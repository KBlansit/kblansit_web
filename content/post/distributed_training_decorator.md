---
title: Why Machine Learning Engineers Should Use Decorators
description: My foray into Keras data distributed training and why we need more decorators
date: "2019-05-02T19:25:30+02:00"
publishDate: "2019-05-02T19:25:30+02:00"
---
#### todo:
# add video
# copy edit
# add active learning part
**Introduction**

I was recently allocated an Azure instance, with 4 K80s for some of my cardiac MRI Autopilot research. This has given me the unique opportunity to experiment with the newer Keras data distributed GPU methods, and think about how to integrate some basic python software engineering best practices into training. In this post, I will first cover how to train with multiple GPUs using distributed data strategy. Then, I will cover how to load a previously trained model, within the same scope (things get tricky here!). Finally, I will show how to clean up the code using decorators, proving more pythonic and extensible code design patterns. I will moreover link to the python scripts that I used for this.

**Versions**

In this post, I will be using the following libraries and version.
- tensorflow-gpu==1.14.0
- keras==2.3.0
- CUDA==10.0
- NVIDIA-Drivers==450.36.06
- Ubuntu==18.04.4

**Distributed Training with Data Parallelism**

To start out with, let’s make a python script for distributed training with data parallelism. I decided to use the MNIST dataset since it’s opensource, free, and lightweight, and allows us to verify everything is working as it should be. I have four primary function here that will form the basis for our work.

[initial_distributed_gpu_training.py](https://github.com/KBlansit/keras_gpu_distributed_example/blob/master/initial_distributed_gpu_training.py)

``` python
def load_minst_data():
    """ loads mnist data
    :returns:
        x_train, y_train, x_test, y_test numpy arrays
    """
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # make into data
    x_train = x_train.reshape(x_train.shape[0],
        IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0],
        IMG_ROWS, IMG_COLS, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # return data
    return x_train, y_train, x_test, y_test
```
This function is self-explanatory and will admit that I borrowed heavily from the Keras MNIST tutorial. However, I personally like to separate my loading data logic into a function.

``` python
def make_datasets(x_input, y_input,
    batch_size=BATCH_SIZE, buffer=None):
    """ returns tensroflow dataset ready for distributed
        training
    :params:
        x_input: training numpy array
        y_input: training numpy array
        batch_size: batch size
        buffer_size: buffer size
    :returns:
        tensorflow dataset
    """
    # if we don't have set buffer size, use all x_input size
    if not buffer: buffer = x_input.shape[0]

    # return dataset
    return tf.data.Dataset.\
        from_tensor_slices((x_input, y_input)).\
        shuffle(buffer).\
        repeat().\
        batch(batch_size, drop_remainder=True)
```
This function takes a paired input and labels (X and Y respectively) and make into a tensorflow Dataset class that can be utilized for distributed training. Circa 7-14-2020 on my specified software setup (see details above), the specified methods called from the Dataset class did not work. After a bit of googling, I saw various suggestions of different combinations of methods to use. Certainly other methods may work better for specific circumstances, and this is something I do intend to venture deeper into when I get a moment. But for now, these methods appear to work for me in this toy dataset as well as my much larger network.

``` python
def calculate_train_and_valid_steps(buffer_size, batch_size):
    """ calculates number of steps needed per batch
    :params:
        batch_size: batch size
        buffer_size: buffer size
    :returns:
        number of steps
    """
    # train number of steps
    if buffer_size % batch_size != 0:
        num_of_steps = buffer_size // batch_size + 1
    else:
        num_of_steps = buffer_size // batch_size

    # find ceiling
    num_of_steps = np.ceil(num_of_steps).astype('int')

    return num_of_steps
```

Since we don’t know the batch size across the multiple GPUs, we instead just want to iterate over a specified number of steps to ensure we have our proper batch size. I like to think of it similar to when your data is from a data generator, where we need to return our data in batch form.

``` python
def cnn_model():
    """ loads a simple cnn model
    :returns:
        Keras model
    """
    # define model
    # conv layers
    inputs = Input(IMG_SHAPE, name = "input")
    conv_1 = Convolution2D(32, KERNEL_SIZE, padding="same",
        activation="relu", name="conv_1")(inputs)
    conv_2 = Convolution2D(64, KERNEL_SIZE, padding="same",
        activation="relu", name="conv_2")(conv_1)
    max_pool = MaxPooling2D(pool_size=(2, 2),
        name="Maxpooling")(conv_2)

    # dense and dropout layers
    drop_1 = Dropout(0.25, name="drop_1")(max_pool)
    flat = Flatten(name="flat")(drop_1)
    dense_1 = Dense(128, name="dense_1")(flat)
    drop_2 = Dropout(0.5, name="drop_2")(dense_1)
    out = Dense(NUM_CLASSES, activation='softmax',
        name="out")(drop_2)

    # define model
    model = Model(inputs=inputs, outputs=out)

    # compile model
    model.compile(
        loss=categorical_crossentropy,
        optimizer=SGD(
            lr=float(LEARN_RATE),
            decay=float(DECAY),
            momentum=MOMENTUM,
        ),
        metrics=['accuracy'],
    )

    # return model
    return model
```

Just a very generic and basic Convolutional Neural Network, and just helps make our code organization better. Plus as you will soon come to see, we can use decorators to modify our model function.

``` python
# load data
x_train, y_train, x_test, y_test = load_minst_data()


# make into tensorflow datasets
# we can use size of datasets since it's <2Gbs
train_buffer_size = x_train.shape[0]
test_buffer_size = x_test.shape[0]

# make datasets
train_dataset = make_datasets(x_train, y_train)
test_dataset = make_datasets(x_test, y_test)

# calculate number of steps
train_parallel_steps = calculate_train_and_valid_steps(
        train_buffer_size,
        BATCH_SIZE,
)
test_parallel_steps = calculate_train_and_valid_steps(
        test_buffer_size,
        BATCH_SIZE,
)


# if we're on the correct VM and are using multi GPUs,
# load scope
if PC == "AiDA-1" and USE_MULTI_GPU:
    # make a learning strategy and open scope for
    # compiling model
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}.".\
        format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = cnn_model()
else:
    model = cnn_model()

```


Now that we have our functions, we can return our data. However, for our model, we need to use the special strategy scope when we compile our data. Prior, we defined above PC to return our machine name (this needs to be changed if you’re not on my VM :wink: ), which may allow more complex behavior if we are moving our model and data training across multiple computers that don’t have multiple GPUs. We additionally have the flag USE_MULTI_GPU to let us quickly turn on and off multi-gpu training.

Finally, let us fit our model, and save the model as a .h5 file. Great! We can now train a model with multiple GPUs.

**Reloading Keras model into Data Distributed Strategy**

Now let us try to load our prior model in distributed GPU mode. There are many reasons to want to pick up a prior trained Keras model, and resume training.  For one, maybe we want to do transfer learning with a prior trained mode? Or maybe our system had a fault halfway through, and we want to resume training without starting all over again. So, lets ignore for a moment our custom scope logic, and just assume for the moment we can do distributed training.

[bad_load_model.py](https://github.com/KBlansit/keras_gpu_distributed_example/blob/master/bad_load_model.py)

``` python
# make a learning strategy and open scope for compiling model
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}.".\
    format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = load_model(PREV_MODEL_PATH)
```

![Drat!](/post/images/distributed_bad_loading_keras_model.png)

:anguished: Huh??? Um, well that’s awkward! Maybe this sorta makes sense, given that we know that the model must be compiled in the strategy scope. After *quite* a bit of googling, I couldn’t find a simple answer. :unamused:

:thinking:However, we know that a Keras model has the methods get_model_weights() and set_model_weight()!
What we can do is load the prior model onto our RAM (rather GPU memory), get the model weights as a list of numpy arrays. We can then recompile a fresh model within the correct strategy scope, and simply load the model weights.


``` python
# load model with cpu
with tf.device('/cpu:0'):
    # load model
    prev_model = load_model(PREV_MODEL_PATH)
    prev_weights = prev_model.get_weights()


# make a learning strategy and open scope for
# compiling model
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}.".\
    format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = cnn_model()
```

![Drat!](/post/images/distributed_good_loading_keras_model.png)

This works! :smile:

However, lets be honest. This code is starting to look messy. And what if we want to keep our custom scope depending on what computer we’re on? That logic would start getting ugly. I think this is an excellent use case for decorators.

**Decorators**

What are decorators? They’re just a simple way of encapsulating custom function logic within another function. I won’t go too much into the specifics of decorators here because I think there’s some great other examples that are worth your time.
[Basic Python Decorators.](https://realpython.com/primer-on-python-decorators/)
[Decorator Functions with Decorator Arguments.](https://www.artima.com/weblogs/viewpost.jsp?thread=240845#decorator-functions-with-decorator-arguments)
[Using Decorators For Fizz Buzz.](https://ryxcommar.com/2019/07/20/fizzbuzz-redux/)

I will admit I put off learning more about them due to the scary syntactic sugar, but after learning that they’re just a function returning a function, they are clearly a useful addition to extend reusable code.

[load_model_multi_gpu.py](https://github.com/KBlansit/keras_gpu_distributed_example/blob/master/load_model_multi_gpu.py)

``` python
def load_model_with_scope(model_func):
    def wrapper(*args, **kwargs):
        # determine if we use multiple GPUs
        if PC == "AiDA-1" and USE_MULTI_GPU:
            # print infromation
            print("Using multi GPU settings on: {}.".\
                format(PC))

            # create a MirroredStrategy and open scope
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                # Everything that creates variables
                # should be under the strategy scope
                # In general this is only model
                # construction & `compile()`
                model = model_func(*args, **kwargs)

                print("Number of devices: {}.".\
                    format(strategy.num_replicas_in_sync))
        # not using multiple GPUs
        else:
            print("Using single GPU settings on: {}.".\
                format(PC))

            model = model_func(*args, **kwargs)
        return model
    return wrapper
```

 Our first decorator will allow us to decorate our model generating functions inside the proper scope. We have internal logic here that allows us to properly scope an arbitrary model function, as well has providing some output information. Finally, we have a model return argument that allows us to get our model back. All this allows us to “hide” the scope code, allowing us to simply extend our code that gets our convolutional neural network.

``` python
def load_model_with_weights(prev_model_path):
    def wrap(model_func):
        def wrapper(*args, **kwargs):
            with tf.device('/cpu:0'):
                # load model
                prev_model = load_model(prev_model_path)
                prev_weights = prev_model.get_weights()

                # clean up so we don't overallocate space
                del prev_model

            # load model and set weights
            model = model_func(*args, **kwargs)
            model.set_weights(prev_weights)

            # message
            print("Set model weights")

            # return
            return model
        return wrapper
    return wrap
```
For our other decorator, we want to be able to load prior model weights. To do so, we need to pass an argument from the top-level function. The way to do this is to apply simply another function to encapsulate our decorator, allowing us to pass that argument within the proper scope. Like above, we load our model, save the weights, make a new model, set the weights, and then return model. To decorate our model with this logic, we just need an additional call to which we pass our model function argument.

However, the logic between the two decorators is abstracted from one another. For loading the model within the load_model_with_weights decorator, we are simply running arbitrary function for the process that gets us our model. We can run it with or without the strategy scope defined in load_model_with_scope depending on how we decorate our model function. The important caveat however is that we first decorate the model function code with the strategy scope, THEN decorate with loading previous weights. If we do it the other way around, our load_model would the potentially be inside a strategy scope, causing error.

**Final Thoughts**

I hope this post helps make the case for machine learning engineers to become more familiar with more advanced topics in python programing. Often time, the emphasis of our field is on developing new and exotic model structures. However, there’s a good case to be made that enhancing machine learning organization can be done with well written code. There’s a certain aesthetic and pride one can get from developing in a world that tries so hard to break design patterns. Not only can we extend logic, we can make code easier to manage, and easier to design experiments.

I do think there are further things I could do to further modularize my code. I certainly could extend decorator logic to extend my functions returning my datasets, so I can automatically create TensorFlow datasets. That said, I hope my code example provides an interesting perspective in how to plan for reusable code in machine learning engineering.
