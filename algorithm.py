import numpy as np 
import torch
import torch.distributions as dist
import math
import copy
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from scipy.special import softmax, logsumexp, log_softmax
import cma


#-------------------------------------------------------------------------
#
#	Prompt Ensembles Algorithm
#
#-------------------------------------------------------------------------
class Ensembles:
	def __init__(self, model_forward_api, intrinsic_dim, popsize, budget, num_samples):
		self.model_forward_api = model_forward_api
		self.intrinsic_dim = intrinsic_dim
		self.popsize = popsize
		self.budget = budget
		self.sample_collections = []
		self.iterations = num_samples

	def sampling(self):
		for i in range(self.iterations):
			model_forward_api = copy.copy(self.model_forward_api)
			cma_opts = {
				'seed': i,
				'popsize': self.popsize,
				'maxiter': self.budget,
				'verbose': -1,
			}
			es = cma.CMAEvolutionStrategy(self.intrinsic_dim * [0], 1, inopts=cma_opts)#sigma=1
			while not es.stop():
				sample_collections = es.ask()
				all_loss, _, _, _ = model_forward_api.eval(sample_collections, parallel=True)
				es.tell(sample_collections, all_loss)
			self.sample_collections.append(model_forward_api.best_prompt)

		weights = np.ones(self.iterations)
		weights = weights / np.sum(weights)
		return self.sample_collections, weights


#-------------------------------------------------------------------------
#
#	Gradient-free Variational Inference Algorithm
#
#-------------------------------------------------------------------------
class CMA_ELBO:
	def __init__(self, model_forward_api, intrinsic_dim, num_samples, variance, seed, popsize, budget, bound, sigma):
		self.model_forward_api = model_forward_api
		self.intrinsic_dim = intrinsic_dim
		self.num_samples = num_samples
		self.mean = torch.zeros(intrinsic_dim)
		self.cov = torch.diag(torch.ones(intrinsic_dim) * variance)
		self.prior = dist.multivariate_normal.MultivariateNormal(self.mean, self.cov)
		self.popsize = popsize
		self.sample_collections = []
		self.cma_opts = {
			'seed': seed,
			'popsize': popsize,
			'maxiter': budget,
			'verbose': -1,
		}
		self.cma_opts['bounds'] = [-1 * bound, 1 * bound]
		self.es = cma.CMAEvolutionStrategy(self.intrinsic_dim * [0] + self.intrinsic_dim * [1], sigma, inopts=self.cma_opts)
		self.ELBO_lambda = 1e-3
		self.ELBO_samples = 10

	def ELBO(self, mean, variance):
		variational_distribution = dist.multivariate_normal.MultivariateNormal(mean, variance)
		theta = [variational_distribution.sample().numpy() for _ in range(self.ELBO_samples)]
		all_loss, _, target,_ = self.model_forward_api.eval(theta, parallel=True)
		weights = np.ones(self.ELBO_samples)
		weights = weights / np.sum(weights)
		val_loss, _ = self.model_forward_api.validation_parallel(theta,weights)
		neg_likelihood = sum(all_loss)*target.size()[0] / len(all_loss)
		kl_divergence = dist.kl_divergence(variational_distribution, self.prior)
		elbo_loss = neg_likelihood + kl_divergence*self.ELBO_lambda
		return elbo_loss, neg_likelihood, kl_divergence, val_loss

	def sampling(self):
		best_val_loss = math.inf
		while not self.es.stop():
			parameter_collections = self.es.ask()
			all_elbo = []
			all_likelihood = []
			for parameter in parameter_collections:
				mean = parameter[:self.intrinsic_dim]
				mean = torch.from_numpy(mean.astype(np.float32))
				variance = parameter[self.intrinsic_dim:]
				variance = torch.from_numpy(variance.astype(np.float32))
				variance = torch.diag(torch.abs(variance))
				elbo_loss, neg_likelihood, kl_divergence, val_loss = self.ELBO(mean, variance)
				if val_loss<best_val_loss:
					best_val_loss = val_loss
					best_mean = mean
					best_variance = variance
				all_elbo.append(elbo_loss.item())
				all_likelihood.append(neg_likelihood)
			self.es.tell(parameter_collections, all_elbo)

		print('best_val_loss',best_val_loss)
		variational_distribution = dist.multivariate_normal.MultivariateNormal(best_mean, best_variance)
		self.sample_collections = [variational_distribution.sample().numpy() for _ in range(self.num_samples)]
		weights = np.ones(self.num_samples)
		weights = weights / np.sum(weights)

		return self.sample_collections, weights


#-------------------------------------------------------------------------
#
#	SBI-based Algorithm (Likelihood-free setting): ABC-SMC Algorithm
#
#-------------------------------------------------------------------------
class ABC_SMC:
	def __init__(self, model_forward_api, intrinsic_dim, num_samples, variance, popsize, weighted=False):
		self.model_forward_api = model_forward_api
		self.intrinsic_dim = intrinsic_dim
		self.num_samples = num_samples
		self.mean = torch.zeros(intrinsic_dim)
		self.cov = torch.diag(torch.ones(intrinsic_dim) * variance)
		self.normal = dist.multivariate_normal.MultivariateNormal(self.mean, self.cov)
		_, _, self.target = self.model_forward_api.eval(self.normal.sample().numpy())
		self.popsize = popsize
		self.N = self.target.size()[0]
		self.weighted = weighted #This flag corresponds to using importance weights or uniform weights
		self.api_calls = 50 #The maximum api_calls for early stopiing the optimization
		# Initilize epsilon as the best accuracy a random prior sample can ahieve
		theta = [self.normal.sample().numpy() for _ in range(self.popsize)]
		all_loss, _, _, all_perf = self.model_forward_api.eval(theta, parallel=True)
		self.epsilon = np.max(all_perf)

	def weighted_variance(self,theta_list_prev, weights_prev):
		theta_all = np.stack(theta_list_prev)
		weights_prev = weights_prev.reshape(-1,1)
		weighted_mean = np.sum(theta_all*weights_prev,axis=0).reshape(1,-1)
		weighted_variance = np.sum(weights_prev*(theta_all-weighted_mean)**2,axis=0)
		weighted_variance = torch.from_numpy(weighted_variance.astype(np.float32))
		weighted_mean = torch.from_numpy(weighted_mean.astype(np.float32))

		return weighted_mean, weighted_variance

	def sampling_weights(self,theta_list_next, disribution_list, weights_prev_log):
		for i in range(self.num_samples):
			prior_distribution = dist.multivariate_normal.MultivariateNormal(self.mean, self.cov)
			prior = prior_distribution.log_prob(torch.from_numpy(theta_list_next[i].astype(np.float32)))
			kernel_prob = np.ones(self.num_samples)
			for j in range(self.num_samples):
				kernel_prob[j] = disribution_list[j].log_prob(torch.from_numpy(theta_list_next[i].astype(np.float32)))
			denominator = logsumexp(weights_prev_log + kernel_prob)
			weights_next_log[i] = prior - denominator
		weights_next = softmax(weights_next_log)
		weights_prev_log = log_softmax(weights_next_log)

		return weights_next, weights_prev_log
		

	def sampling(self):
		theta_list_prev = []
		theta_list_next = []
		weights_prev = np.ones(self.num_samples)
		weights_prev = weights_prev/np.sum(weights_prev)
		weights_prev_log = np.log(weights_prev)
		best_val_perf = 0
		flag_initial = True
		
		while self.epsilon<=1:
			print('epsilon', self.epsilon)
			count=0
			count_api_cal = 0
			stop_flag = False
			best_loss = math.inf
			if flag_initial:
				while count < self.num_samples:
					theta = [self.normal.sample().numpy() for _ in range(self.popsize)]
					_, _, _, all_perfs = self.model_forward_api.eval(theta, parallel=True)
					index_list = [idx for idx, perf in enumerate(all_perfs) if perf >= self.epsilon]
					for item in index_list:
						theta_list_prev.append(theta[item])
						count += 1
						if count == self.num_samples:
							break
				self.epsilon = self.epsilon + 1/self.N
				# Validation
				theta_list_val = theta_list_prev
				val_loss, val_perf = self.model_forward_api.validation_parallel(theta_list_val, weights_prev)
				if val_perf >= best_val_perf:
					best_val_perf = val_perf
					best_theta_list = theta_list_prev
					best_weight = weights_prev
				flag_initial = False
			else:
				if self.weighted:
					proposal_cov = torch.diag(torch.ones(self.intrinsic_dim) * 10)
				else:
					adaptive_mean, adaptive_variance = self.weighted_variance(theta_list_prev, weights_prev)
					variance_mean = torch.mean(adaptive_variance)
					proposal_cov = torch.diag(adaptive_variance/variance_mean * 10) 
				#Mixture proposal distribution
				disribution_list = [dist.multivariate_normal.MultivariateNormal(torch.from_numpy(item.astype(np.float32)), proposal_cov) for item in theta_list_prev]
				while count < self.num_samples:
					theta_next = []
					count_pop = 0
					while count_pop<self.popsize:
						choice_index = np.random.choice(np.arange(self.num_samples), p=weights_prev)
						proposal_dist = disribution_list[choice_index]
						theta_next.append(proposal_dist.sample().numpy())
						count_pop +=1
					_, _, _, all_perfs = self.model_forward_api.eval(theta_next, parallel=True)
					index_list = [idx for idx, perfs in enumerate(all_perfs) if perfs >= self.epsilon]
					for item in index_list:
						theta_list_next.append(theta_next[item])
						count += 1
						if count == self.num_samples:
							break

					#Stop sampling if optimization cannot progress further(no sample can satisfy the tolerance)
					count_api_cal += 1
					if count==0 and count_api_cal>self.api_calls:
						stop_flag = True
						break
				#Trigger for early stop		
				if stop_flag:
					break

				self.epsilon = self.epsilon + 1/self.N
				#Calculate the sampling weights corresponding to each sample theta
				if self.weighted:
					weights_next, weights_prev_log  = self.sampling_weights(theta_list_next, disribution_list, weights_prev_log)
				weights_next = np.ones(self.num_samples)
				weights_next = weights_next/np.sum(weights_next)
				theta_list_prev = theta_list_next
				theta_list_next = []
				weights_prev = weights_next
				
				#Validation, pick the optimal collection of theta
				theta_list_val = theta_list_prev
				val_loss, val_perf = self.model_forward_api.validation_parallel(theta_list_val, weights_next)
				if val_perf >= best_val_perf:
					best_val_perf = val_perf
					best_theta_list = theta_list_prev
					best_weight = weights_next

		return best_theta_list, best_weight



#-------------------------------------------------------------------------
#
#	SBI-based Algorithm (Likelihood-free setting): SNPE
# 	Reference: https://www.mackelab.org/sbi/
#
#-------------------------------------------------------------------------
class SBI_neural:
	def __init__(self, model_forward_api, intrinsic_dim, num_samples, variance, num_labels, popsize):
		self.model_forward_api = model_forward_api
		self.intrinsic_dim = intrinsic_dim
		self.num_samples = num_samples
		self.mean = torch.zeros(intrinsic_dim)
		self.cov = torch.diag(torch.ones(intrinsic_dim) * variance)
		self.prior = dist.multivariate_normal.MultivariateNormal(self.mean, self.cov)
		self.popsize = popsize
		self.num_labels = num_labels
		theta = self.prior.sample().numpy()
		_, _, target = self.model_forward_api.eval(theta)
		one_hot = torch.zeros(target.size()[0], self.num_labels)
		one_hot[torch.arange(target.size()[0]), target] = 1
		self.observation = one_hot.view(1, -1)

	def simulation(self, iteration, proposal, num_simulations):
		theta_collection = torch.zeros((num_simulations, self.intrinsic_dim))
		observation_collection = torch.zeros((num_simulations, self.observation.size()[1]))
		count = 0
		best_index = 0
		best_perf =0
		while count < num_simulations:
			if iteration == 0:
				theta = [proposal.sample().numpy() for _ in range(self.popsize)]
			else:
				theta_list = proposal.sample((self.popsize,), show_progress_bars=True)
				theta = [item.numpy() for item in theta_list]

			_, all_output, _, all_perfs = self.model_forward_api.eval(theta, parallel=True)
			for item in range(self.popsize):
				theta_collection[count] = torch.from_numpy(theta[item].astype(np.float32))
				target = all_output[item].argmax(dim=-1)
				one_hot = torch.zeros(target.size()[0], self.num_labels)
				one_hot[torch.arange(target.size()[0]), target] = 1
				observation_collection[count] = one_hot.view(1, -1)
				if all_perfs[item] >=best_perf:
					best_index = count
					best_perf = all_perfs[item]
				count += 1
				if count % 100 == 0:
					print('Already collected num of sample:', count)
				if count == num_simulations:
					break

		return theta_collection, observation_collection, best_index

	def sampling(self):
		density_estimator_build_fun = posterior_nn(model='maf', hidden_features=60, num_transforms=10, num_components=10, device="cuda:0")
		inference = SNPE(prior=self.prior, density_estimator= density_estimator_build_fun)
		num_rounds = 5
		best_val_perf = 0
		proposal = self.prior
		weights = np.ones(self.num_samples)
		weights = weights / np.sum(weights)
		for i in range(num_rounds):
			print('current iteration', i)
			#Collect the simulated observation(LLM prediction) and corresponding parameter(prompt theta)
			theta, observation, best_index = self.simulation(i, proposal, num_simulations=1000)
			#Infer the posterior distribution of theta
			density_estimator = inference.append_simulations(theta, observation, proposal=proposal).train()
			posterior = inference.build_posterior(density_estimator)
			proposal = posterior.set_default_x(observation[best_index])
			#Sampling new collection of parameter from posterior distribution 
			samples = posterior.sample((self.num_samples,), x= observation[best_index])
			sample_collections = [item.numpy() for item in samples]
			#Validation, pick the optimal collection of theta
			val_loss, val_perf = self.model_forward_api.validation_parallel(sample_collections,weights)
			if val_perf >= best_val_perf:
				print('best_val_perf', val_perf.item())
				best_val_perf = val_perf
				best_theta_list = sample_collections

		return best_theta_list, weights

