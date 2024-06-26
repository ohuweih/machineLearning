# Loading in the mnist dataset
from keras.datasets.mnist import load_data
from matplotlib import pyplot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Dropout, Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
#from keras.utils.vis_utils import plot_model
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy import vstack
from numpy.random import randn
from numpy.random import randint

##########################################################################################
######################### This block is to visualize the type of data ####################
######################### we are training against ########################################
##########################################################################################

# Load images in to memory
#(trainX, trainy), (testX, testy) = load_data()

# Show training and testing example size and pixel size
#print ('Train', trainX.shape, trainy.shape)
#print('Test', testX.shape, testy.shape)

#plot images from the training set
#for i in range(25):
    # define subplot
#    pyplot.subplot( 5, 5, 1 + i)
    # Turn off axis
#    pyplot.axis('off')
    # plot raw pixel data of training image
    #pyplot.imshow(trainX[i], cmap='gray')

    # plot raw pixel data of training image in reverse (white background black number)
#    pyplot.imshow(trainX[i], cmap='gray_r')
#pyplot.savefig('plotted_training_set_with_reverse_gray_scale.png')

# define model
#model = discriminator()
# summarize the model
#model.summary()
# plot the model
#plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)


# Defining the discriminator, in_shape of pixel size of our images
def defineDiscriminator(input_shape=(28,28,1)):
    model = Sequential()
    # Going from 28 * 28 pixels to 14 * 14
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    # going from 14 * 14 pixels to 7 * 7
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def loadRealSamples():
    # Load mnist training dataset
    (trainX, _), (_, _) = load_data()
    # Expand images from 2d to 3d, Adding a channel dimension
    X = expand_dims(trainX, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')
    # Scale from [0, 255] to [0, 1]
    X = X / 255.0
    return X

# Select real samples
def generateRealSamples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    #retrieve selected images
    X = dataset[ix]
    #generate 'real' class labels
    y = ones((n_samples, 1))
    return X, y

# Generate number of fake samples with class labels
def generateFakeSamples(g_model, latent_dim, n_samples):
    # Generate points in latent space
    x_input = generateLatentPoints(latent_dim, n_samples)
    # Predict outputs
    X = g_model.predict(x_input)
    # Generate 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

#####################################################################################################################
#####################################################################################################################
################################# This and the train GAN below was the first way we trained #########################
################################# It didnt really work, There is a train function that combined the 2 below##########
#####################################################################################################################
#####################################################################################################################
# Train the discriminator model
#def trainDiscriminator(model, dataset, n_iter=100, n_batch=256):
#    half_batch = int(n_batch /2)
    # enumerate epochs
#    for i in range(n_iter):
        # Get random real samples
#        X_real, y_real = generateRealSamples(dataset, half_batch)
        #update discriminator on real samples
#        _, real_acc = model.train_on_batch(X_real, y_real)
        # generate fake examples
#        X_fake, y_fake = generateFakeSamples(half_batch)
        # Update discriminator on fake samples
#        _, fake_acc = model.train_on_batch(X_fake, y_fake)
        # summarixe performance
#        print('>%d real = %.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

def defineGenerator(latent_dim):
    model = Sequential()
    # the foundation for a 7X7 image
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # Upsample to 14X14
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # Upsample to 28X28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

# Generate points in latent space as input for the generator
def generateLatentPoints(latent_dim, n_samples):
    # Generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    #reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Define the combined generator and discriminator model, for updating the generator
def defineGAN(g_model, d_model):
    # Make weights in the discriminator not trainable
    d_model.trainable = False
    # Connect the models
    model = Sequential()
    # Add generator
    model.add(g_model)
    # Add discriminator
    model.add(d_model)
    # Compile model
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model



##############################################################################################
##############################################################################################
############################### This is how we trained our models the first time #############
############################### It didnt really work #########################################
##############################################################################################
##############################################################################################

# Train the composite model
#def trainGAN(gan_model, latent_dim, n_epochs=100, n_batch=256):
    # Manually enumerate epochs
#    for i in range(n_epochs):
        # Prepare points in latent space as input for the generator
#        x_gan = generateLatentPoints(latent_dim, n_batch)
        # Create inverted labes for the fake samples
#        y_gan = ones((n_batch,1 ))
        # Update the generator via the discriminator's error
#        gan_model.train_on_batch(x_gan, y_gan)

# Train the the generator and discriminator, This will combine the trainGAN and trainDiscriminator functions.
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # Manually enumerate epochs
    for i in range(n_epochs):
        # Enumerate batches over the training set
        for j in range(batch_per_epoch):
            # Get real samples
            X_real, y_real = generateRealSamples(dataset, half_batch)
            # Generate fake samples
            X_fake, y_fake = generateFakeSamples(g_model, latent_dim, half_batch)
            # Create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # Update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            # Prepare points in latent space as input for the generator
            X_gan = generateLatentPoints(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # Update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # Summarize loss on this batch
            print('>%d, %d%d, d=%.3f, g=%.3f' % (i+1, j+1, batch_per_epoch, d_loss, g_loss))
        if (i+1) % 10 == 0:
            evaluatePerformance(i, g_model, d_model, dataset, latent_dim)

# Evaluate the discriminator, Plot images, save generator model
def evaluatePerformance(epoch, g_model, d_model, dataset, latent_dim, n_samples=128):
    # Get real samples
    X_real, y_real = generateRealSamples(dataset, n_samples)
    # Evaluate discriminator on real samples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # Generate fake samples
    X_fake, y_fake = generateFakeSamples(g_model, latent_dim, n_samples)
    # Evaluate discriminator on fake samples
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)
    # Print discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # Save plot
    savePlot(X_fake, epoch)
    # Save the generator model weights file
    filename = 'generator_model_%03d.keras' % (epoch + 1)
    g_model.save(filename)

# Create and save generated images (reversed grayscale)
def savePlot(x_fake, epoch, n=10):
    for i in range(n * n):
        # Define Subplot. We did this in the begining of this file too
        pyplot.subplot(n, n, 1 + i)
        # Turn of axis
        pyplot.axis('off')
        # plot raw pixel data in reversed grayscale
        pyplot.imshow(x_fake[i, :, :, 0], cmap='gray_r')
        # save plot to file
        filename = 'generated_plot_e%03d.png' % (epoch+1)
        pyplot.savefig(filename)
        pyplot.close()


# Size of latent space
latent_dim = 100
# Create the discriminator
d_model = defineDiscriminator()
# Create the generatpor
g_model = defineGenerator(latent_dim)
# Create the GAN
gan_model = defineGAN(g_model, d_model)
# Load image data
dataset = loadRealSamples()
# Train model
train(g_model, d_model, gan_model, dataset, latent_dim)
# Summarize gan model
gan_model.summary()


# plot gan model
#plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)
