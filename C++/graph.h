#pragma once
#include <vector>
#include <utility>
#include <unordered_map>
#include "kdtree.hpp"
#include <boost/functional/hash.hpp>
#include <boost/algorithm/clamp.hpp>
#include <boost/random/discrete_distribution.hpp>

#define M_PI 3.14159265358979323846

typedef std::pair<int, int> vec2;
typedef std::pair<vec2, vec2> edge;

struct pair_hash {
	std::size_t operator() (const vec2& p) const {
		std::size_t seed = 0;

		boost::hash_combine(seed, p.first);
		boost::hash_combine(seed, p.second);

		return seed;
	}
};

struct pairOfpairHash {
	std::size_t operator()(const edge& p) const {
		std::size_t seed = 0;

		boost::hash_combine(seed, p.first.first);
		boost::hash_combine(seed, p.first.second);
		boost::hash_combine(seed, p.second.first);
		boost::hash_combine(seed, p.second.second);

		return seed;
	}
};

float gaussian(float x, float sigma) {
	return exp(-pow(x, 2) / (2 * pow(sigma, 2)));
}

class graph
{
public:
	// node list
	std::vector<vec2> nodes;

	// edge map
	std::unordered_map<vec2, std::vector<vec2>, pair_hash> neighbor_map;

	// edge weight map
	std::unordered_map<edge, double, pairOfpairHash> edge_weight_map;

	// edge intersection map
	std::unordered_map<edge, std::vector<edge>, pairOfpairHash> edge_intersection_map;
	
	// constructor
	graph() {}

	// constructor with parameters
	graph(std::vector<vec2> nodes, 
		std::unordered_map<edge, std::vector<edge>, pairOfpairHash> edge_intersection_map, 
		std::unordered_map<edge, double, pairOfpairHash> edge_weight_map, 
		std::unordered_map<vec2, std::vector<vec2>, pair_hash> neighbor_map) {
		this->nodes = nodes;
		this->edge_intersection_map = edge_intersection_map;
		this->edge_weight_map = edge_weight_map;
		this->neighbor_map = neighbor_map;
	}

	// destructor
	~graph() {}

	// Getters
	float get_a_pore_dist() { return a_pore_dist; }
	float get_angle() { return angle; }
	float get_a_s() { return a_s; }
	float get_a_dist() { return a_dist; }
	float get_a_cont() { return a_cont; }
	float get_a_cross() { return a_cross; }
	float get_a_sim() { return a_sim; }
	float get_a_deposit() { return a_deposit; }
	float get_a_decay() { return a_decay; }
	
	float get_a_wrinkle_width() { return a_wrinkle_width; }
	float get_a_pore_width() { return a_pore_width; }
	float get_d_min() { return d_min; }
	float get_a_blend() { return a_blend; }
	float get_a_skew() { return a_skew; }
	float get_a_cushion() { return a_cushion; }
	float get_f_perturb() { return f_perturb; }
	float get_s_perturb() { return s_perturb; }
	float get_f_noise() { return f_noise; }
	float get_s_noise() { return s_noise; }
	float get_a_scale() { return a_scale; }

	// Setters
	void set_a_pore_dist(float a_pore_dist) { this->a_pore_dist = a_pore_dist; }
	void set_angle(float angle) { this->angle = angle; }
	void set_a_s(float a_s) { this->a_s = a_s; }
	void set_a_dist(float a_dist) { this->a_dist = a_dist; }
	void set_a_cont(float a_cont) { this->a_cont = a_cont; }
	void set_a_cross(float a_cross) { this->a_cross = a_cross; }
	void set_a_sim(float a_sim) { this->a_sim = a_sim; }
	void set_a_deposit(float a_deposit) { this->a_deposit = a_deposit; }
	void set_a_decay(float a_decay) { this->a_decay = a_decay; }

	void set_a_wrinkle_width(float a_wrinkle_width) { this->a_wrinkle_width = a_wrinkle_width; }
	void set_a_pore_width(float a_pore_width) { this->a_pore_width = a_pore_width; }
	void set_d_min(float d_min) { this->d_min = d_min; }
	void set_a_blend(float a_blend) { this->a_blend = a_blend; }
	void set_a_skew(float a_skew) { this->a_skew = a_skew; }
	void set_a_cushion(float a_cushion) { this->a_cushion = a_cushion; }
	void set_f_perturb(float f_perturb) { this->f_perturb = f_perturb; }
	void set_s_perturb(float s_perturb) { this->s_perturb = s_perturb; }
	void set_f_noise(float f_noise) { this->f_noise = f_noise; }
	void set_s_noise(float s_noise) { this->s_noise = s_noise; }
	void set_a_scale(float a_scale) { this->a_scale = a_scale; }

	// Random walk probability functions

	// Probability of relevant orientation
	float orientation_probability(float angle, float angel_sub, int num_pis) {
		return std::min({std::abs(angle - angel_sub + 0 * M_PI),
			std::abs(angle - angel_sub + num_pis * M_PI),
			std::abs(angle - angel_sub - num_pis * M_PI)});
	}
	float compute_p_pref(const vec2& current, const vec2& next) {
		std::pair<float, float> preferredOrientation = std::make_pair(cos(this->angle), sin(this->angle));
		std::pair<float, float> currentOrientation = std::make_pair(next.first - current.first, next.second - current.second);
		// Normalize
		float norm = sqrt(pow(currentOrientation.first, 2) + pow(currentOrientation.second, 2));
		currentOrientation.first /= norm;
		currentOrientation.second /= norm;
		// Compute dot product
		float dot = preferredOrientation.first * currentOrientation.first + preferredOrientation.second * currentOrientation.second;

		// Compute angle between preferred and current orientation
		float ang = acos(dot);

		// Gaussian distribution mean
		float sigma = 0.3;

		// Compute probability, std::exp(-std::pow(orientation_probability(ang,1), 2.0) / (std::pow(theta, 2.0)));
		float prob = this->a_s + gaussian(orientation_probability(ang,this->angle ,1), sigma);

		return boost::algorithm::clamp(prob, 0.0, 1.0);
	}
	float compute_p_cross(const vec2& current, const vec2& next) {
		edge currentEdge = std::make_pair(current, next);

		float sum = 0.0;
		for (int i = 0; i < this->edge_intersection_map[currentEdge].size(); i++) {
			sum += this->edge_weight_map[this->edge_intersection_map[currentEdge][i]];
		}

		return sum;
	}
	float compute_p_cont(const vec2& current, const vec2& next) {
		float currentAngle = std::atan2(next.second - current.second, next.first - current.first);

		edge currentEdge = std::make_pair(current, next);
		float p_cont = 0.0;
		for (int i = 0; i < this->neighbor_map[current].size(); i++) {
			vec2 neighbor = std::make_pair(
				this->neighbor_map[current][i].first,
				this->neighbor_map[current][i].second
			);
			edge neighborEdge = std::make_pair(current, neighbor);
			if (currentEdge != neighborEdge) {
				float neighborAngle = std::atan2(neighbor.second - current.second, neighbor.first - current.first);
				float orientation_prob = orientation_probability(currentAngle, (neighborAngle + M_PI), 2);
				float orientation_prob_sim = orientation_probability(currentAngle, neighborAngle, 2);
				p_cont += this->edge_weight_map[neighborEdge] * this->a_cont * (gaussian(orientation_prob, 0.3))
					- this->a_sim * gaussian(orientation_prob_sim, 0.3);

			}
		}
		return p_cont;
	}
	float compute_distance_penalty(const vec2& current, const vec2& next){
		float dist = sqrt(pow(next.first - current.first, 2) + pow(next.second - current.second, 2));
		return 1.0 / pow(dist, this->a_dist);
	}
	float wrinkle_probability(const vec2& current, const vec2& next) {
		float distance_penalty = compute_distance_penalty(current, next);
		float p_pref = compute_p_pref(current, next);
		float p_cross = compute_p_cross(current, next);
		float p_cont = compute_p_cont(current, next);

		float prob = distance_penalty * (p_cont + (1.0 - this->a_cont) * p_pref - this->a_cross * p_cross);
		return std::max({0.0f, prob});
	}

	// Weighed sampling
	int sample(std::vector<float> probabilities) {
		std::mt19937 gen(std::random_device{}());
		boost::random::discrete_distribution<> dist(probabilities);
		return dist(gen);
	}

	// Random walk functions
	// Parallelable random walk
	std::unordered_map<edge, double, pairOfpairHash> random_walk(vec2& start, int numIterations) {
		std::unordered_map<edge, double, pairOfpairHash> weight_map;
		vec2 current = start;
		std::mt19937 gen(std::random_device{}());
		for (int i = 0; i < numIterations; i++) {
			// Get neighbors
			std::vector<vec2> neighbors = this->neighbor_map[current];
			// Compute probabilities
			std::vector<float> probabilities;
			float sum = 0.0;
			for (int j = 0; j < neighbors.size(); j++) {
				probabilities.push_back(wrinkle_probability(current, neighbors[j]));
				sum += probabilities[j];
			}

			if (sum == 0.0) {
				// sample a random node from node list
				std::uniform_int_distribution<int> dist(0, this->nodes.size() - 1);
				int index = dist(gen);
				current = this->nodes[index];
				continue;
			}

			// Normalize probabilities
			for (int j = 0; j < probabilities.size(); j++) {
				probabilities[j] /= sum;
			}
			// Sample next point
			int next_index = sample(probabilities);
			vec2 next = neighbors[next_index];

			// Update weight map
			edge currentEdge = std::make_pair(current, next);
			if (weight_map.find(currentEdge) == weight_map.end()) {
				weight_map[currentEdge] = 0.0;
			}
			double curEdgeWeight = weight_map[currentEdge];
			curEdgeWeight = curEdgeWeight + this->dt * this->a_deposit * (1.0 - curEdgeWeight);
			weight_map[currentEdge] = curEdgeWeight;

			// Update current point
			current = next;

		}

		// Apply global decay
		for (auto& edgeIter : weight_map) {
			float factor = 1.0 - this->dt * this->a_decay * (numIterations / weight_map.size());
			weight_map[edgeIter.first] = edgeIter.second * factor;
		}

		return weight_map;
	}

	// Random walk in-place
	void random_walk_in_place(vec2& start, int numIterations) {
		vec2 current = start;
		std::mt19937 gen(std::random_device{}());
		for (int i = 0; i < numIterations; i++) {
			// Get neighbors
			std::vector<vec2> neighbors = this->neighbor_map[current];
			// Compute probabilities
			std::vector<float> probabilities;
			float sum = 0.0;
			for (int j = 0; j < neighbors.size(); j++) {
				probabilities.push_back(wrinkle_probability(current, neighbors[j]));
				sum += probabilities[j];
			}

			if (sum == 0.0) {
				// sample a random node from node list
				std::uniform_int_distribution<int> dist(0, this->nodes.size() - 1);
				int index = dist(gen);
				current = this->nodes[index];
				continue;
			}

			// Normalize probabilities
			for (int j = 0; j < probabilities.size(); j++) {
				probabilities[j] /= sum;
			}
			// Sample next point
			int next_index = sample(probabilities);
			vec2 next = neighbors[next_index];

			// Update weight map
			edge currentEdge = std::make_pair(current, next);
			if (this->edge_weight_map.find(currentEdge) == this->edge_weight_map.end()) {
				this->edge_weight_map[currentEdge] = 0.0;
			}
			double curEdgeWeight = this->edge_weight_map[currentEdge];
			curEdgeWeight = curEdgeWeight + this->dt * this->a_deposit * (1.0 - curEdgeWeight);
			this->edge_weight_map[currentEdge] = curEdgeWeight;

			// Update current point
			current = next;

		}

		// Apply global decay
		for (auto& edgeIter : this->edge_weight_map) {
			float factor = 1.0 - this->dt * this->a_decay * (numIterations / this->edge_weight_map.size());
			this->edge_weight_map[edgeIter.first] = edgeIter.second * factor;
		}
	}

private:
	// Average pore distance
	float a_pore_dist = 3.0;
	// Primary orientation [0, 2pi]
	float angle = 0;
	// Orientation uniformity [0, 1]
	float a_s = 0.0;
	// Distance exponent 
	float a_dist = 5.0;
	// Continuation reward
	float a_cont = 0.0;
	// Crossing penalty
	float a_cross = 3.0;
	// Similarity penalty
	float a_sim = 0.0;
	// weight deposit strength
	float a_deposit = 1.0;
	// weight decay strength
	float a_decay = 0.5;

	// Wrinkle width
	float a_wrinkle_width = 0.6;
	// Pore width
	float a_pore_width = 0.6;
	// Min. pore strength
	float d_min = 0.0;
	// Blending from edges to pores
	float a_blend = 0.0;
	// Pore skew strength
	float a_skew = 0.0;
	// Cushioning effect
	float a_cushion = 0.0;
	// Wrinkle pertubation frequency
	float f_perturb = 1500;
	// Wrinkle pertubation strength
	float s_perturb = 0.0;
	// Additive noise frequency
	float f_noise = 0.0;
	// Additive noise strength
	float s_noise = 0.0;
	// Global displacement scaling
	float a_scale = 0.5;

	// Delta time
	float dt = 0.03;

};

