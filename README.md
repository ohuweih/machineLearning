# machineLearning
Python Machine Learning code for learning how ML works and hopefully get something with ML to work and to work well

# Image Generation
python3 createImage.py

This will take in a dataset of images and try to generate a like image. It has a generator that rates itself on how well it creates the image, a discriminator that rates itself on how well it determans whether an image is AI generated or not. A GAN (Generative adversarial network) which takes the generate and discriminator and runs them against each other

This does not work. My first attemped at making a GAN failed lol. Keeping this around as notes for future refrance A working example of this is V2


# Image Generation V2

python3 createImage_V2.py

In this version of image generation we dont train the discrimiator. Instead we feed the discriminator with 'real' (images from our dataset) and 'fake' (images we generate from latent space) images then update the generator via the composite model. 

We save the generator weights every 10 epoch (10 batches of training). on save we run an evaluation which includes saving a generated image. 

Composite model explained 
https://alan-turing-institute.github.io/MLJ.jl/v0.16/composing_models/

Needed to run this code


pip3.10 install numpy
pip3.10 install matplotlib
pip3.10 install pydot
pip3.10 install keras
python3.10 -m pip install tensorflow[and-cuda]  ## this is for GPU based training
python3.10 -m pip install tensorflow            ## this is for CPU based training
pip3.10 install tensorrt
pip3.10 install pydot
pip3.10 install keras
apt install graphviz


#Image Discription
python3 feedImage.py

This will take an image and return 3 things it thinks it can see in the image. This works as is but its not very good. It seems this is only as good as the dataset you feed it for its training/comperaion. 


