import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

# load and preprocess the MNIST dataset (This can be replace with any dataset)
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype(numpy.float32) - 127.5) / 127.5
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# Define the generator model
def build_generator():
    generator = Sequential()
    generator.add(Dense(256, input_dim=100))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(2048))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    generator.add(Dense(28 * 28 * 1, activation='tanh'))
    generator.add(Reshape((28, 28, 1 )))
    return generator

# Define the Discriminator model
def build_discriminator():
    discriminator = Sequential()
    discriminator.add(Flatten(input_shape=(28, 28, 1)))

    discriminator.add(Dense(2048))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1024))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(1, activation='sigmoid'))
    return discriminator

#Build and Compile GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    generator.trainable = True
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001, beta_1=0.5))
    return gan

#Train GAN
def train_gan(generator, discriminator, gan, epochs=30000, batch_size=128):
    # Try to load already done training. 
    try:
        print("Loading previous training")
        generator.load_weights('generator_weights.h5')
        discriminator.load_weights('discriminator_weights.h5')
    except OSError as e:
        print("No previous training to load")


    for epoch in range(epochs):
        noise = numpy.random.normal(0, 1, size=(batch_size, 100))
        generated_images = generator.predict(noise)
        image_batch = x_train[numpy.random.randint(0, x_train.shape[1], size=batch_size)]
        X = numpy.concatenate([image_batch, generated_images])
        y_dis = numpy.zeros(2 * batch_size)
        y_dis[:batch_size] = 0.9
        y_gen = numpy.ones(batch_size)

        #Train discriminator
        d_loss = discriminator.train_on_batch(X, y_dis)

        #Train generator
        g_loss = gan.train_on_batch(noise, y_gen)

        #print progress, save generated images at intervals, and save weights
        if epoch % 100 == 0:
            print(f"Epoch {epoch} [D loss: {d_loss:.4f}] [G loss:{g_loss:.4f}]")
            save_generated_images(generator, epoch)
            if discriminator.trainable == True:
                discriminator.save_weights('discriminator_weights.h5')
                print("Saved discriminator weights")
            elif generator.trainable == True:
                generator.save_weights('generator_weights.h5')
                print("Saved generator weights")
            else:
                print("Nothing is saving bro")
# Save generated images
def save_generated_images(generator, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = numpy.random.normal(0, 1, size=(examples, 100))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"generated_image_epch_{epoch}.png")

#Build models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

#compile generator and discriminator
generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001, beta_1=0.5))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001, beta_1=0.5))

#Train the Gan 
train_gan(generator, discriminator, gan)

