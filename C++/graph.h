#pragma once
#include <vector>
#include <utility>
#include <unordered_map>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/functional/hash.hpp>
#include <boost/algorithm/clamp.hpp>
#include <boost/algorithm/minmax.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/geometry/geometries/box.hpp>
#include "kdtree.hpp"
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define M_PI 3.14159265358979323846
#define GRID_SIZE 1024
#define N 10

using point = boost::geometry::model::point<int, 2, boost::geometry::cs::cartesian>;
using segment = boost::geometry::model::segment<point>;
using box = boost::geometry::model::box<point>;

bool operator==(const point& p1, const point& p2) {
	return boost::geometry::get<0>(p1) == boost::geometry::get<0>(p2) &&
		boost::geometry::get<1>(p1) == boost::geometry::get<1>(p2);
}

bool operator==(const segment& l1, const segment& l2) {
	return boost::geometry::get<0, 0>(l1) == boost::geometry::get<0, 0>(l2) &&
		boost::geometry::get<0, 1>(l1) == boost::geometry::get<0, 1>(l2) &&
		boost::geometry::get<1, 0>(l1) == boost::geometry::get<1, 0>(l2) &&
		boost::geometry::get<1, 1>(l1) == boost::geometry::get<1, 1>(l2);
}

struct point_equal {
	bool operator() (const point& p1, const point& p2) const {
		return boost::geometry::get<0>(p1) == boost::geometry::get<0>(p2) &&
			boost::geometry::get<1>(p1) == boost::geometry::get<1>(p2);
	}
};

struct segment_equal {
	bool operator() (const segment& l1, const segment& l2) const {
		return l1.first == l2.first && l1.second == l2.second;
	}
};

struct pair_hash {
	std::size_t operator() (const point& p) const {
		std::size_t seed = 0;

		boost::hash_combine(seed, boost::geometry::get<0>(p));
		boost::hash_combine(seed, boost::geometry::get<1>(p));

		return seed;
	}
};

struct pairOfpairHash {
	std::size_t operator()(const segment& l) const {
		std::size_t seed = 0;

		boost::hash_combine(seed, boost::geometry::get<0, 0>(l));
		boost::hash_combine(seed, boost::geometry::get<0, 1>(l));
		boost::hash_combine(seed, boost::geometry::get<1, 0>(l));
		boost::hash_combine(seed, boost::geometry::get<1, 1>(l));

		return seed;
	}
};

float gaussian(float x, float sigma) {
	return exp(-pow(x, 2) / (2 * pow(sigma, 2)));
}

float fnv1Hash(int x) {
	const uint32_t prime = 0x01000193; //   16777619
	int ret = -2128831035;

	int b0 = (x & 0x000000ff) >> 0;
	int b1 = (x & 0x0000ff00) >> 8;
	int b2 = (x & 0x00ff0000) >> 16;
	int b3 = (x & 0xff000000) >> 24;

	ret = (ret * prime) ^ b0;
	ret = (ret * prime) ^ b1;
	ret = (ret * prime) ^ b2;
	ret = (ret * prime) ^ b3;

	return ret;
}

point gradient(point p) {
	int x = fnv1Hash(boost::geometry::get<0>(p));
	int y = fnv1Hash(x + boost::geometry::get<1>(p));

	return point(std::sin(x + y), std::sin(y + y));
}

class graph
{
public:
	// node list
	std::vector<point> nodes;
	// edge list 
	std::vector<segment> edges;

	// node neighbor map
	std::unordered_map<point, std::vector<point>, pair_hash, point_equal> neighbor_map;

	// edge weight map
	std::unordered_map<segment, double, pairOfpairHash, segment_equal> edge_weight_map;

	// edge intersection map
	std::unordered_map<segment, std::vector<segment>, pairOfpairHash, segment_equal> edge_intersection_map;
	
	
	// constructor
	graph() {}

	// constructor with parameters
	graph(std::vector<point> nodes, 
		std::unordered_map<segment, double, pairOfpairHash, segment_equal> edge_weight_map,
		std::unordered_map<point, std::vector<point>, pair_hash, point_equal> neighbor_map) {
		this->nodes = nodes;
		this->edge_weight_map = edge_weight_map;
		this->neighbor_map = neighbor_map;

		// construct edge list
		for (auto& edge : edge_weight_map) {
			edges.push_back(edge.first);
		}

		// initialize edge intersection map
		for (auto& edge : edges) {
			edge_intersection_map[edge] = std::vector<segment>();
		}

		this->findIntersectionsParallel();

		std::cout << "number of edges: " << edges.size() << std::endl;
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

	// gradient w frequency controlled
	point gradient_controlled(point p) {
		p = point(boost::geometry::get<0>(p) * this->f_perturb, boost::geometry::get<1>(p) * this->f_perturb);

		int x = fnv1Hash(boost::geometry::get<0>(p));
		int y = fnv1Hash(x + boost::geometry::get<1>(p));

		return point(std::sin(x+y) * this->s_perturb, std::sin(y+y) * this->s_perturb);
	}


	// parallel find intersections
	void intersectionWorker(int start, int end) {
		// print thread id
		// std::cout << "Thread " << std::this_thread::get_id() << " started at " << start << " and ended at " << end << std::endl;
		for (int i = start; i < end; i++) {
			std::vector<segment> intersections;
			for (int j = 0; j < this->edges.size(); j++) {
				if (!(this->edges[i] == this->edges[j])) {
					if (boost::geometry::intersects(this->edges[i], this->edges[j])) {
						intersections.push_back(this->edges[j]);
					}
				}
			}
			this->edge_intersection_map[this->edges[i]] = intersections;
		}
	}

	// create edge intersection map
	void findIntersectionsParallel() {
		int threads = std::thread::hardware_concurrency();
		std::vector<std::thread> workers;
		int start = 0;
		int end = 0;
		int step = this->edges.size() / threads;

		// create threads
		for (int i = 0; i < threads; i++) {
			start = i * step;
			end = start + step;
			if (i == threads - 1) {
				end = this->edges.size();
			}
			workers.push_back(std::thread(&graph::intersectionWorker, this, start, end));
		}

		// wait for all threads to finish
		for (auto& worker : workers) {
			worker.join();
		}

		// print stats
		std::cout << "Number of intersections: " << this->edge_intersection_map.size() << std::endl;
	}
	
	// Random walk probability functions
	// Probability of relevant orientation
	float orientation_probability(float angle, float angel_sub, int num_pis) {
		return std::min({std::abs(angle - angel_sub + 0 * M_PI),
			std::abs(angle - angel_sub + num_pis * M_PI),
			std::abs(angle - angel_sub - num_pis * M_PI)});
	}
	
	float compute_p_pref(const point& current, const point& next) {
		// Compute preferred orientation
		std::pair<float, float> preferredOrientation = std::make_pair(cos(this->angle), sin(this->angle));
		std::pair<float, float> currentOrientation = std::make_pair(boost::geometry::get<0>(next) - boost::geometry::get<0>(current), 
			boost::geometry::get<1>(next) - boost::geometry::get<1>(current));
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

	float compute_p_cross(const point& current, const point& next) {
		segment currentEdge = segment(current, next);

		float sum = 0.0;
		if (this->edge_intersection_map.find(currentEdge) == this->edge_intersection_map.end()) {
			if (this->edge_intersection_map.find(segment(next, current)) == this->edge_intersection_map.end()) {
				return 0.0;
			}
			else {
				currentEdge = segment(next, current);
			}
		}
		for (int i = 0; i < this->edge_intersection_map[currentEdge].size(); i++) {
			sum += this->edge_weight_map[this->edge_intersection_map[currentEdge][i]];
		}

		return sum;
	}

	float compute_p_cont(const point& current, const point& next) {
		float currentAngle = std::atan2(boost::geometry::get<1>(next) - boost::geometry::get<1>(current),
			boost::geometry::get<0>(next) - boost::geometry::get<0>(current));

		segment currentEdge = segment(current, next);
		float p_cont = 0.0;
		if (this->neighbor_map.find(current) == this->neighbor_map.end()) {
			return 0.0;
		}
		for (int i = 0; i < this->neighbor_map[current].size(); i++) {
			point neighbor = this->neighbor_map[current][i];
			segment neighborEdge = segment(current, neighbor);
			if (!(currentEdge == neighborEdge)) {
				float neighborAngle = std::atan2(boost::geometry::get<1>(neighbor) - boost::geometry::get<1>(current),
										boost::geometry::get<0>(neighbor) - boost::geometry::get<0>(current));
				float orientation_prob = orientation_probability(currentAngle, (neighborAngle + M_PI), 2);
				float orientation_prob_sim = orientation_probability(currentAngle, neighborAngle, 2);

				if (this->edge_weight_map.find(neighborEdge) == this->edge_weight_map.end()) {
					if (this->edge_weight_map.find(segment(neighbor, current)) == this->edge_weight_map.end()) {
						continue;
					}
					else {
						neighborEdge = segment(neighbor, current);
					}
				}
				p_cont += this->edge_weight_map[neighborEdge] * this->a_cont * (gaussian(orientation_prob, 0.3))
					- this->a_sim * gaussian(orientation_prob_sim, 0.3);

			}
		}
		return p_cont;
	}
	
	float compute_distance_penalty(const point& current, const point& next){
		float dist = boost::geometry::distance(current, next);
		return 1.0 / pow(dist, this->a_dist);
	}

	float wrinkle_probability(const point& current, const point& next) {
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
	std::unordered_map<segment, double, pairOfpairHash, segment_equal> random_walk(int numIterations) {
		std::unordered_map<segment, double, pairOfpairHash, segment_equal> weight_map;
		
		std::mt19937 gen(std::random_device{}());
		std::uniform_int_distribution<int> dist(0, this->nodes.size() - 1);
		point current = this->nodes[dist(gen)];
		for (int i = 0; i < numIterations; i++) {
			// Get neighbors
			std::vector<point> neighbors = this->neighbor_map[current];
			// Compute probabilities
			std::vector<float> probabilities;
			float sum = 0.0;
			for (int j = 0; j < neighbors.size(); j++) {
				probabilities.push_back(wrinkle_probability(current, neighbors[j]));
				sum += probabilities[j];
			}

			if (sum == 0.0) {
				// sample a random node from node list
				
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
			point next = neighbors[next_index];

			// Update weight map
			segment currentEdge = segment(current, next);
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
	void random_walk_in_place(int numIterations) {
		
		std::cout << "Number of edges: " << this->edge_weight_map.size() <<std::endl;
		std::mt19937 gen(std::random_device{}());
		std::uniform_int_distribution<int> dist(0, this->nodes.size() - 1);
		point current = this->nodes[dist(gen)];

		for (int i = 0; i < numIterations; i++) {
			// Get neighbors
			std::vector<point> neighbors = this->neighbor_map[current];
			// Compute probabilities
			std::vector<float> probabilities;
			float sum = 0.0;
			for (int j = 0; j < neighbors.size(); j++) {
				probabilities.push_back(wrinkle_probability(current, neighbors[j]));
				sum += probabilities[j];
			}

			if (sum == 0.0) {
				// sample a random node from node list
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
			point next = neighbors[next_index];

			// Update weight map
			segment currentEdge = segment(current, next);
			if (this->edge_weight_map.find(currentEdge) == this->edge_weight_map.end()) {
				if (this->edge_weight_map.find(segment(next,current)) != this->edge_weight_map.end()) {
					currentEdge = segment(next, current);
				}
				else {
					this->edge_weight_map[currentEdge] = 0.0;
				}
				
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

	// Random walk worker function
	void parallelWorker(int numIterations, std::unordered_map<segment, double, pairOfpairHash, segment_equal>& weight, int id){
		weight = this->edge_weight_map;
		std::mt19937 gen(std::random_device{}());
		std::uniform_int_distribution<int> dist(0, this->nodes.size() - 1);
		point current = this->nodes[dist(gen)];

		std::cout << "thread " << id << " started..." << std::endl;
		for (int i = 0; i < numIterations; i++) {
			// Get neighbors
			std::vector<point> neighbors = this->neighbor_map[current];
			// Compute probabilities
			std::vector<float> probabilities;
			float sum = 0.0;
			for (int j = 0; j < neighbors.size(); j++) {
				probabilities.push_back(wrinkle_probability(current, neighbors[j]));
				sum += probabilities[j];
			}

			if (sum == 0.0 || probabilities.size() == 0) {
				// sample a random node from node list
				int index = dist(gen);
				current = this->nodes[index];
				continue;
			}

			// Normalize probabilities
			for (int j = 0; j < probabilities.size(); j++) {
				probabilities[j] /= sum;
			}
			// Sample next point
			int maxIndex = 0;
			float maxProb = 0.0;
			for (int j = 0; j < probabilities.size(); j++) {
				if (probabilities[j] > maxProb) {
					maxProb = probabilities[j];
					maxIndex = j;
				}
			}
			// int next_index = maxIndex;

			int next_index = sample(probabilities);
			point next = neighbors[next_index];

			// Update weight map
			segment currentEdge = segment(current, next);
			if (weight.find(currentEdge) == weight.end()) {
				if (weight.find(segment(next, current)) != weight.end()) {
					currentEdge = segment(current, next);
				}
				else {
					weight.emplace(currentEdge, 0.0);
				}
			}
			double curEdgeWeight = weight[currentEdge];
			curEdgeWeight = curEdgeWeight + this->dt * this->a_deposit * (1.0 - curEdgeWeight);
			weight[currentEdge] = curEdgeWeight;

			// Update current point
			current = this->nodes[dist(gen)];
		}
	}

	// Random walk parallel
	void randomWalkParallel(int iterations) {
		int numThreads = std::thread::hardware_concurrency();
		std::vector<std::thread> threads;
		std::vector<std::unordered_map<segment, double, pairOfpairHash, segment_equal>> weight_maps(numThreads);

		// Start threads
		std::cout << "Starting " << numThreads << " threads..." << std::endl;
		for (int i = 0; i < numThreads; i++) {
			threads.push_back(std::thread(&graph::parallelWorker, this, iterations / numThreads, std::ref(weight_maps[i]), i));
		}

		// Wait for threads to finish
		std::cout << "Waiting for threads to finish..." << std::endl;
		for (int i = 0; i < numThreads; i++) {
			threads[i].join();
			std::cout << "Thread " << i << " finished." << std::endl;
		}

		// Combine weight maps
		std::cout << "Combining weight maps..." << std::endl;
		for (int i = 0; i < numThreads; i++) {
			for (auto& edgeIter : weight_maps[i]) {
				if (this->edge_weight_map.find(edgeIter.first) == this->edge_weight_map.end()) {
					this->edge_weight_map[edgeIter.first] = 0.0;
				}
				this->edge_weight_map[edgeIter.first] += edgeIter.second;
			}
		}

		// Apply global decay
		std::cout << "Applying global decay..." << std::endl;
		for (auto& edgeIter : this->edge_weight_map) {
			float factor = 1.0 - this->dt * this->a_decay * (iterations / this->edge_weight_map.size());
			this->edge_weight_map[edgeIter.first] = edgeIter.second * factor;
		}
	}

	// Draw output graph to image in opencv
	void draw_graph() {
		// Create image
		cv::Mat image(GRID_SIZE, GRID_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));

		// Draw edges with thickness and color based on weight
		for (auto& edgeIter : this->edge_weight_map) {
			point start = edgeIter.first.first;
			point end = edgeIter.first.second;
			double weight = edgeIter.second;
			cv::Scalar color = cv::Scalar(255, 255, 255) - cv::Scalar(255, 255, 255) * weight;

			cv::line(image,
				cv::Point(boost::geometry::get<0>(start), boost::geometry::get<1>(start)), 
				cv::Point(cv::Point(boost::geometry::get<0>(end), boost::geometry::get<1>(end))),
				color, std::exp(weight));
		}
		// Draw nodes
		for (auto& node : this->nodes) {
			cv::circle(image, cv::Point(boost::geometry::get<0>(node), boost::geometry::get<1>(node)), 1, cv::Scalar(0, 0, 0), 1);
		}
		// display image
		cv::imshow("image", image);
		cv::waitKey(0);
		// Save image
		cv::imwrite("graph.png", image);
	}

	double shape_f(int x) {
		return std::pow(-(1 - 2 * x / 3), 3);
	}

	double mellowmax(std::vector<double> vals) {
		int n_hat = 16;
		int beta = 20;
		double sum = 0.0;
		for (int i = 0; i < vals.size(); i++) {
			sum += std::exp(beta * vals[i]);
		}
		return (1.0 / beta) * std::log((1.0/n_hat) * (n_hat - vals.size() + sum));
	}

	// Draw output as displacement map
	void draw_displacement() {
		// calculate node weights
		std::unordered_map<point, double, pair_hash, point_equal> node_weight_map;
		for (auto& node : this->nodes) {
			double w_max = 0.0;
			double w_sum = 0.0;
			for (int i = 0; i < this->neighbor_map[node].size(); i++) {
				segment edge = segment(node, this->neighbor_map[node][i]);
				if (this->edge_weight_map.find(edge) != this->edge_weight_map.end()) {
					if (this->edge_weight_map.find(segment(this->neighbor_map[node][i], node)) != this->edge_weight_map.end()) {
						edge = segment(this->neighbor_map[node][i], node);
					}
				}
				else {
					continue;
				}
				double w = this->edge_weight_map[edge];
				w_sum += w;
				w_max = std::max(w_max, w);
			}
			node_weight_map[node] = std::max({ w_max + this->d_min, w_max + this->a_blend * (w_sum - w_max) });
		}

		// calculate edge displacement
		// create a 2d vector for result image
		std::vector<std::vector<double>> displacementMap(GRID_SIZE, std::vector<double>(GRID_SIZE));


		// create a 3d vector for each pixel (x, y, weights) to store weights
		std::vector<std::vector<std::vector<double>>> displacementMapVals(GRID_SIZE, std::vector<std::vector<double>>(GRID_SIZE, std::vector<double>()));
		

		// wrinkle width
		float w_radius = 1.5; 

		// calculate displacement for each edge
		std::cout << "Calculating displacement for edge..." << std::endl;
		for (auto& edge : this->edges) {
			float wrinkle_width = this->a_wrinkle_width * this->edge_weight_map[edge];
			// std::cout << "wrinkle_width: " << wrinkle_width << std::endl;
			if (wrinkle_width < 0.01) {
				continue;
			}

			// get AABB of edge
			box edge_box = box();
			boost::geometry::envelope(edge, edge_box);
			
			// loop through all pixels in AABB
			float x_min = boost::geometry::get<0>(edge_box.min_corner());
			float x_max = boost::geometry::get<0>(edge_box.max_corner());
			float y_min = boost::geometry::get<1>(edge_box.min_corner());
			float y_max = boost::geometry::get<1>(edge_box.max_corner());

			// print range
			// std::cout << "x_min: " << x_min << " x_max: " << x_max << " y_min: " << y_min << " y_max: " << y_max << std::endl;

			// actual radius
			float r = wrinkle_width * w_radius;

			for (int x = x_min; x < x_max; x++) {
				for (int y = y_min; y < y_max; y++) {
					// calculate distance to edge
					point p = point(x, y);
					double dist = boost::geometry::distance(p, edge);
					if (dist > r) {
						continue;
					}
					point perturbedPoint = this->gradient_controlled(p);
					int xp = boost::geometry::get<0>(perturbedPoint) + x;
					int yp = boost::geometry::get<1>(perturbedPoint) + y;
					xp = std::min(std::max(xp, 0), GRID_SIZE - 1);
					yp = std::min(std::max(yp, 0), GRID_SIZE - 1);

					// calculate displacement
					double displacement = this->shape_f(dist) * wrinkle_width;
					displacementMapVals[xp][yp].push_back(displacement);
					displacementMapVals[x][y].push_back(displacement);

				}
			}

		}

		//// calculate displacement for each node, create cushioning effect
		//std::cout << "Calculating displacement for node..." << std::endl;
		//for (auto& node : this->nodes) {
		//	float pore_width = this->a_pore_width * node_weight_map[node];
		//	
		//	// for neighboring edges
		//	for (int i = 0; i < this->neighbor_map[node].size(); i++) {
		//		// get direction to neighbor
		//		point neighbor = this->neighbor_map[node][i];
		//		point normalizedDirection = point(boost::geometry::get<0>(neighbor) - boost::geometry::get<0>(node), boost::geometry::get<1>(neighbor) - boost::geometry::get<1>(node));
		//		double length = boost::geometry::distance(node, neighbor);
		//		boost::geometry::divide_value(normalizedDirection, length);

		//		// loop length, add node weight to displacement map along the edge
		//		for (int i = 0; i < length; i++) {
		//			point p = point(boost::geometry::get<0>(node) + i * boost::geometry::get<0>(normalizedDirection), boost::geometry::get<1>(node) + i * boost::geometry::get<1>(normalizedDirection));
		//			int xP = boost::geometry::get<0>(p);
		//			int yP = boost::geometry::get<1>(p);

		//			// calculate displacement
		//			double displacement = -node_weight_map[node] * std::exp(-this->a_cushion * (i+1));
		//			displacementMapVals[xP][yP].push_back(displacement);
		//		}


		//	}

		//}

		// mellowmax each pixel of the displacement map
		std::cout << "Mellowmaxing..." << std::endl;
		for (int x = 0; x < GRID_SIZE; x++) {
			for (int y = 0; y < GRID_SIZE; y++) {
				displacementMap[x][y] = this->mellowmax(displacementMapVals[x][y]);
			}
		}
		


		// convert to opencv image
		std::cout << "Converting to opencv image..." << std::endl;
		cv::Mat displacementImage = cv::Mat(GRID_SIZE, GRID_SIZE, CV_64F);

		for (int x = 0; x < GRID_SIZE; x++) {
			for (int y = 0; y < GRID_SIZE; y++) {
				displacementImage.at<double>(x, y) = displacementMap[x][y];
			}
		}

		
		int scale = 4;
		cv::Mat scaledDisplacement = cv::Mat(GRID_SIZE * scale, GRID_SIZE * scale, CV_64F);

		for (int x = 0; x < GRID_SIZE * scale; x++) {
			for (int y = 0; y < GRID_SIZE * scale; y++) {
				scaledDisplacement.at<double>(x, y) = displacementImage.at<double>(x % GRID_SIZE, y % GRID_SIZE);
			}
		}

		
		// read a target exr image
		const char* input = "cavity.exr";
		float* out; // width * height * RGBA
		int width = GRID_SIZE * scale;
		int height = GRID_SIZE * scale;
		const char* err = NULL; // or nullptr in C++11

		int ret = LoadEXR(&out, &width, &height, input, &err);

		if (ret != TINYEXR_SUCCESS) {
			if (err) {
				fprintf(stderr, "ERR : %s\n", err);
				FreeEXRErrorMessage(err); // release memory of error message.
			}
		}

		// add displacement to target image
		std::cout << "Adding displacement to target image..." << std::endl;
		cv::Mat displacedImage = cv::Mat(width, height, CV_64F);
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				// RGBA
				int nextIndex = (x * width + y) * 4;
				displacedImage.at<double>(x, y) = out[nextIndex] + scaledDisplacement.at<double>(x, y) * 2;
			}
		}
		//// display image
		//cv::imshow("displacement", displacedImage);
		//cv::waitKey(0);


		// write to image

		// 1. Normalize the image values to the 0-1 range
		cv::Mat normalizedImage;
		cv::normalize(displacedImage, normalizedImage, 0.0, 1.0, cv::NORM_MINMAX);

		// 2. Convert the floating-point image to 8-bit
		cv::Mat outputImage;
		normalizedImage.convertTo(outputImage, CV_8UC3, 255.0);

		// 3. Save the image
		cv::imwrite("output.png", outputImage);

		//// save to file
		//std::cout << "Saving to file..." << std::endl;
		//std::ofstream myfile;
		//myfile.open("displacementMap.txt");
		//for (int x = 0; x < GRID_SIZE * 4; x++) {
		//	for (int y = 0; y < GRID_SIZE * 4; y++) {
		//		// RGBA
		//		myfile << displacedImage.at<double>(x, y) << " ";
		//	}
		//	myfile << "\n";
		//}

		free(out);
	}
private:
	// Average pore distance
	float a_pore_dist = 3.0;
	// Primary orientation [0, 2pi]
	float angle = -0.16 * M_PI;
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
	float d_min = 0.6;
	// Blending from edges to pores
	float a_blend = 1.0;
	// Pore skew strength
	float a_skew = 0.0;
	// Cushioning effect
	float a_cushion = 0.8;
	// Wrinkle pertubation frequency
	float f_perturb = 1500;
	// Wrinkle pertubation strength
	float s_perturb = 0.0;
	// Additive noise frequency
	float f_noise = 3000.0;
	// Additive noise strength
	float s_noise = 0.35;
	// Global displacement scaling
	float a_scale = 0.5;

	// Delta time
	float dt = 0.03;

};

