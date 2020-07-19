import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.__version__
# Part 1 - Data Preprocessing
# Generating images for the Training set
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Generating images for the Test set
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

# Creating the Training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creating the Test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(training_set,
                  steps_per_epoch = 32,
                  epochs = 63,
                  validation_data = test_set,
                  validation_steps = 32)

#printing the model summary
cnn.summary()

#Saving the model
cnn.save(r"F:\myModel")

#Loading the trained model 
load = tf.keras.models.load_model(r"F:\myModel")
#prediction function
def predictME():
    X = test_datagen.flow_from_directory('dataset/ha',
                                         target_size = (64, 64),
                                         batch_size = 32,
                                         class_mode = 'binary')
    p = load.predict(X)
    p = ['dog' if n >= 0.5 else 'cat' for n in p]
    return p

c = predictME()


