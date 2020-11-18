# Cannoy

CNN + Annoy

**Convolution Neural Network**
ImageNet Classification with Deep Convolutional Neural Networks
https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

**Approximate Neareat Neighbors Oh Yeah**
https://github.com/spotify/annoy

**FastAI**
Deep learning Framework 
https://github.com/fastai/fastai


## Deep learning for similar image search

**Resnet** 
Neural network
https://arxiv.org/pdf/1512.03385.pdf
https://arxiv.org/pdf/1603.05027.pdf

## Making model
|dataset|image count|
|-------|-----------|
|train|25450|
|val|8443|
|test|8482|

## Epoch 10times
|epoch|train_loss|valid_loss|accuracy|top_k_accuracy|top_k_accuracy|time|
|-----|----------|----------|--------|--------------|--------------|----|
|0|1.585172|0.986633|0.613109|0.940705|0.983261|02:05|
|1|1.098105|0.785079|0.644347|0.974537|0.995638|02:04|
|2|0.824332|0.665368|0.673582|0.979960|0.998114|02:05|
|3|0.730807|0.635824|0.680891|0.985147|0.998585|02:06|
|4|0.666662|0.645393|0.673936|0.986090|0.998468|02:06|
|5|0.622534|0.597592|0.696923|0.987033|0.998468|02:05|
|6|0.593684|0.561883|0.698102|0.992102|0.998939|02:05|
|7|0.569229|0.547308|0.689025|0.992573|0.999175|02:05|
|8|0.535214|0.547930|0.687846|0.991395|0.998821|02:05|
|9|0.529549|0.547237|0.688318|0.991512|0.998821|02:06|

## Make tree with annoy

	from annoy import AnnoyIndex
	import random

	f = 40
	t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
	for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

	t.build(10) # 10 trees
	t.save('test.ann')

	#...
	
	u = AnnoyIndex(f, 'angular')
	u.load('test.ann') # super fast, will just mmap the file
	print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors


## Search time

### without annoy
**2.2232918739318848 s**

### with annoy
**0.004664897918701172 s**
