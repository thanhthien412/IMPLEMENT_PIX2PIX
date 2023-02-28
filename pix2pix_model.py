from tqdm import tqdm
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle


def define_discriminator(image_shape=(256,256,3)):
    
	# weight initialization
	init = RandomNormal(stddev=0.02) #As described in the original paper
    # use to create initial kernel
	# source image input
	in_src_image = Input(shape=image_shape)  #Image we want to convert to another image
	# target image input
	in_target_image = Input(shape=image_shape)  #Image we want to generate after training. 
    
	# concatenate images, channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
    
	# C64: 4x4 kernel Stride 2x2
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128: 4x4 kernel Stride 2x2
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256: 4x4 kernel Stride 2x2
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512: 4x4 kernel Stride 2x2 
    # Not in the original paper. Comment this block if you want.
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer : 4x4 kernel but Stride 1x1
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d) # shape 16x16x1 
	patch_out = Activation('sigmoid')(d)
 
    
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model


def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g



# define a decoder block to be used in generator
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g


def define_generator(image_shape=(256,256,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model: C64-C128-C256-C512-C512-C512-C512-C512
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 64)
	e3 = define_encoder_block(e2, 128)
	e4 = define_encoder_block(e3, 256)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model: CD512-CD512-CD512-C512-C256-C128-C64
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 256, dropout=False)
	d5 = decoder_block(d4, e3, 128, dropout=False)
	d6 = decoder_block(d5, e2, 64, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(image_shape[2], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7) #Modified 
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False 
            
	# define the source image
	in_src = Input(shape=image_shape)
	# suppy the image as input to the generator 
	gen_out = g_model(in_src)
	# supply the input image and generated image as inputs to the discriminator
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and disc. output as outputs
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
    
    #Total loss is the weighted sum of adversarial loss (BCE) and L1 loss (MAE) coefficient is in paper
	model.compile(loss=['binary_crossentropy', 'mae'], 
               optimizer=opt, loss_weights=[1,100])
	return model


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples,verbose=0)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y



# generate samples and save as a plot and save the model
#GAN models do not converge, we just want to find a good balance between
#the generator and the discriminator. Therefore, it makes sense to periodically
#save the generator model and check how good the generated image looks. 
def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples*2 + i)
		plt.axis('off')
		plt.imshow(X_realB[i])
	# save plot to file
	filename1 = 'train/plot_%06d.png' % (step+1)
	plt.savefig(filename1)
	plt.close()
	g_model.compiled_metrics==None
	# save the generator model
	filename2 = 'weight/model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# train pix2pix models
def train(d_model, g_model, gan_model, dataset,state=0,n_epochs=100,n_batch=2):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]

	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	Train_sample,Train_target=dataset
	num=len(trainA)
	if(state!=0):
		g_model.load_weights('weight/model_%06d.h5'%state)
		print('successfully update--- weight/model_%06d.h5'%state)


	for epochs in range(state,n_epochs):
		Train_sample,Train_target=shuffle(Train_sample,Train_target,random_state=epochs)
		totall_loss1=0.0
		totall_loss2=0.0
		totall_loss_g=0.0
 
		for batch in tqdm(range(bat_per_epo)):
			y_real = ones((2, n_patch, n_patch, 1))
			# generate a batch of fake samples
			Train_fake, y_fake = generate_fake_samples(g_model,Train_sample[batch*n_batch:batch*n_batch+n_batch],n_patch)
			# update discriminator for real samples
			d_loss1=d_model.train_on_batch([Train_sample[batch*n_batch:batch*n_batch+n_batch],Train_target[batch*n_batch:batch*n_batch+n_batch]],y_real)
			# update discriminator for fake samples
			d_loss2=d_model.train_on_batch([Train_sample[batch*n_batch:batch*n_batch+n_batch],Train_fake],y_fake)
			# update the generator
			g_loss,_,_=gan_model.train_on_batch(Train_sample[batch*n_batch:batch*n_batch+n_batch],[y_real,Train_target[batch*n_batch:batch*n_batch+n_batch]])
			#summarize
			totall_loss1+=d_loss1
			totall_loss2+=d_loss2
			totall_loss_g+=g_loss

		print('epoch: %d, d1[%.3f] d2[%.3f] g[%.3f]' % (epochs+1, totall_loss1/num,totall_loss2/num, totall_loss_g/num))
		
		if(epochs+1) %10 ==0:
			summarize_performance(epochs,g_model,dataset)


def predict(g_model, dataset,n_sample=3):
    Val_sample,Val_target=dataset
    Val_fake=[]
    batch=int(len(Val_sample)/n_sample)
    print(batch)
    for i in range(batch):
        Val_fake.append(generate_fake_samples(g_model,Val_sample[i*n_sample:i*n_sample+n_sample],1)[0])

    Val_fake=np.concatenate(Val_fake,axis=0)
    Val_sample= (Val_sample+1)/2.0
    Val_target= (Val_target+1)/2.0
    Val_fake= (Val_fake+1)/2.0
    
    for i in tqdm(range(batch)):
        for j in range(n_sample):
            plt.subplot(3,n_sample,1+j)
            plt.axis('off')
            plt.imshow(Val_sample[i*n_sample+j])
            
        for j in range(n_sample):
            plt.subplot(3,n_sample,1+j+n_sample)
            plt.axis('off')
            plt.imshow(Val_fake[i*n_sample+j])

        for j in range(n_sample):
            plt.subplot(3,n_sample,1+j+n_sample*2)
            plt.axis('off')
            plt.imshow(Val_target[i*n_sample+j])
        
        filename1='result/plot_%06d.png'%(i+1)
        plt.savefig(filename1)
        plt.close
    