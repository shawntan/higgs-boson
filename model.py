import math
import theano
import theano.tensor as T
import numpy         as np
import utils         as U
import numpy as np
np.random.seed(12345)
from data import *
import cPickle as pickle
import sys,random
from theano.tensor.shared_randomstreams import RandomStreams
#from numpy_hinton import print_arr

def build_network(input_size,hidden_size):
	srng = RandomStreams(seed=12345)

	X = T.dmatrix('X')
	W_input_to_hidden1  = U.create_shared(U.initial_weights(input_size,hidden_size))
	b_hidden1 = U.create_shared(U.initial_weights(hidden_size))
	W_hidden1_to_output = U.create_shared(U.initial_weights(hidden_size))
	b_output = U.create_shared(U.initial_weights(1)[0])
	
	def network(training):
		hidden1 = T.dot(X,W_input_to_hidden1) + b_hidden1
		hidden1 = hidden1 * (hidden1 > 0)
		if training:
			hidden1 = hidden1 * srng.binomial(size=(hidden_size,),p=0.5)
		else:
			hidden1 = 0.5 * hidden1
		output = T.nnet.sigmoid(T.dot(hidden1,W_hidden1_to_output) + b_output)
		return output
	
	parameters = [
		W_input_to_hidden1,
		b_hidden1,
		W_hidden1_to_output,
		b_output
	]

	return X,network(True),network(False),parameters

def build_cost(output,test_output,params):
	Y = T.bvector('Y')
	weight = T.dvector('weight')
	b_r = 10

#	l1 = 1e-8*sum( T.sum(abs(p)) for p in params )
#	pred_s = theano.printing.Print('Shape')(pred_s)
	def ams(pred_s):
		s = T.sum(weight * Y * pred_s)
		b = T.sum(weight * (1-Y) * pred_s)
		ams_score = T.sqrt(2*((s + b + b_r) * T.log(1 + s/(b + b_r)) - s))
		return ams_score
	def neg_log_ams_approx(pred_s):
		s = T.sum(weight * Y * pred_s)
		b = T.sum(weight * (1-Y) * pred_s)
		log_ams_approx = 0.5*T.log(b + 10) - T.log(s + 0.01)
		return log_ams_approx

#	log_loss = -T.mean(Y*T.log(output) + (1-Y)*T.log(1-output))
		
	return Y,weight,-ams(output), ams(test_output>0.5)

if __name__ == '__main__':
	params_file = sys.argv[2]
	data,labels,weights,_, feature_names = load_data(sys.argv[1])
	input_width = data.shape[1]
	data = U.create_shared(data)
	labels = U.create_shared(labels,dtype=np.int8)
	weights = U.create_shared(weights)

	X,output,test_output,parameters = build_network(input_width,1024)
	Y, w, cost, ams = build_cost(output,test_output,parameters)
	gradients = T.grad(cost,wrt=parameters)

	eps = T.dscalar('eps')
	mu  = T.dscalar('mu')
	deltas = [ U.create_shared(np.zeros(p.get_value().shape)) for p in parameters ]
	delta_nexts = [ mu*delta + eps*grad for delta,grad in zip(deltas,gradients) ]
	delta_updates = [ (delta, delta_next) for delta,delta_next in zip(deltas,delta_nexts) ]
	param_updates = [ (param, param - delta_next) for param,delta_next in zip(parameters,delta_nexts) ]
	
	batch_size = 50000
	training_set = 200000
	batch = T.lvector('batch')
	train = theano.function(
			inputs=[batch,eps,mu],
			outputs=cost,
			updates=delta_updates + param_updates,
			givens={
				X: data[batch],
				Y: labels[batch],
				w: weights[batch],
			}
		)
	test = theano.function(
			inputs=[],
			outputs=ams,
			givens={
				X: data[training_set:],
				Y: labels[training_set:],
				w: weights[training_set:],
			}
		)
	
	best_ams = 0
	for b in xrange(10000):
		unseen = np.ones(training_set,dtype=np.int8)
		while unseen.sum() >= batch_size:
			sample = np.random.choice(training_set,batch_size,p=unseen/float(unseen.sum()))
			unseen[sample] = 0
			train(sample,0.1,0.9)
		ams = test()
		if best_ams < ams:
			with open(params_file,'wb') as f:
				pickle.dump({
					'feature_names': feature_names,
					'parameters': [ p.get_value() for p in parameters ]
				},f)
			best_ams = ams
		print ams


