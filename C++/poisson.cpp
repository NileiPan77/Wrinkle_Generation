
#include <iostream>
#include <vector>
#include <random>
#include <iterator>
#include <fstream>
#include <string>
#include <set>
#include <ctime>
#include "kdtree.hpp"
#include "graph.h"

using namespace cv;

bool is_validPoint(int x, int y, int grid_size, bool *grid[GRID_SIZE]) {
    // check if the point is in the grid
    if (x >= 0 && x < grid_size && y >= 0 && y < grid_size) {
        // check if the point is in the grid
        if (grid[x][y] == 1) {
            return false;
        }

        // initialize the radius
        int r = (N - 1) / 2;

        // check if there exist a point in its nxn neighborhood
        for (int i = -r; i <= r; i++) {
            for (int j = -r; j <= r; j++) {
                if (x + i >= 0 && x + i < grid_size && y + j >= 0 && y + j < grid_size) {
                    if (grid[x + i][y + j] == 1) {
                        return false;
                    }
                }
            }
        }
    }
    else {
        return false;
    }

    return true;
}


std::vector<point> poissonSample() {
    // initialize a grid
    bool** grid = new bool* [GRID_SIZE];
    for (int i = 0; i < GRID_SIZE; i++) {
        grid[i] = new bool[GRID_SIZE];
    }

    // initialize the grid with 0
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            grid[i][j] = 0;
        }
    }

    // initialize the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // initialize the first sample
    float x = dis(gen);
    float y = dis(gen);
    int i = floor(x * GRID_SIZE);
    int j = floor(y * GRID_SIZE);
    grid[i][j] = 1;

    // initialize the active list
    std::vector<point> active_list;
    active_list.push_back(point(i, j));

    // initialize the final sample list
    std::vector<point> final_list;

    // insert the first sample into the final list
    final_list.push_back(point(i, j));

    // insert the first sample into the active list
    active_list.push_back(point(i, j));

    // initialize max number of attempts
    int max_attempts = 30;

    // initialize the radius
    float radius = (N - 1) / 2;

    // generate the samples
    while (!active_list.empty()) {
        // randomly select a sample from the active list
        int index = floor(dis(gen) * active_list.size());
        point sample = active_list[index];

        // initialize the flag
        bool flag = false;

        // Try up to max_attempts times to find a candidate
        for (size_t i = 0; i < max_attempts; i++) {
            // generate a random angle
            float angle = dis(gen) * 2 * M_PI;

            // generate a random radius
            float r = dis(gen) * radius + radius;

            // calculate the candidate
            int x = floor(boost::geometry::get<0>(sample) + r * cos(angle));
            int y = floor(boost::geometry::get<1>(sample) + r * sin(angle));

            // check if the candidate is valid
            if (is_validPoint(x, y, GRID_SIZE, grid)) {
                // insert the candidate into the active list
                active_list.push_back(point(x, y));

                // insert the candidate into the final list
                final_list.push_back(point(x, y));

                // insert the candidate into the grid
                grid[x][y] = 1;

                // set the flag to true
                flag = true;

                // break the loop
                break;
            }
        }

        // if the flag is false, remove the sample from the active list
        if (!flag) {
            active_list.erase(active_list.begin() + index);
        }
    }
    
    // free the memory
    for (int i = 0; i < GRID_SIZE; i++) {
        delete[] grid[i];
    }
    delete[] grid;

    return final_list;
}


int main(int argc, char* argv[]) {

    time_t start, end;
    start = clock();

    std::cout << "Poisson Disk Sampling" << std::endl;

    std::vector<point> final_list = poissonSample();
    
    
    // Use kdtree to find k-nearest neighbors for edge connection
    Kdtree::KdNodeVector nodes;
    for (int i = 0; i < final_list.size(); i++) {
        std::vector<double> point(2);
        point[0] = boost::geometry::get<0>(final_list[i]);
        point[1] = boost::geometry::get<1>(final_list[i]);
		nodes.push_back(Kdtree::KdNode(point));
	}
    
    // build the kdtree
    Kdtree::KdTree tree(&nodes);
    
    // k nearest neighbors
    int k = 7;

    // edge map
    std::unordered_map<point, std::vector<point>, pair_hash, point_equal> neighbor_map;

    // edge weight map
    std::unordered_map<segment, double, pairOfpairHash, segment_equal> edge_weight_map;

    
    // connect the edges
    for (int i = 0; i < final_list.size(); i++) {
        
        std::vector<point> edge_map_vec;
        Kdtree::KdNodeVector result;
        std::vector<double> p(2);
        p[0] = boost::geometry::get<0>(final_list[i]);
        p[1] = boost::geometry::get<1>(final_list[i]);
        tree.k_nearest_neighbors(p, k, &result);
        // edge_map.insert(std::make_pair(final_list[i], result));

        // Remove the first element in the result (which is the point itself)
        result.erase(result.begin());
        for (int j = 0; j < result.size(); j++) {
            edge_map_vec.push_back(point(result[j].point[0], result[j].point[1]));
            segment newEdge = segment(final_list[i], point(result[j].point[0], result[j].point[1]));
            segment newEdge2 = segment(point(result[j].point[0], result[j].point[1]), final_list[i]);
            if (edge_weight_map.find(newEdge) != edge_weight_map.end() || edge_weight_map.find(newEdge2) != edge_weight_map.end()) {
                continue;
			}
            else {
				edge_weight_map[newEdge] = 0;
			}
		}
        neighbor_map[final_list[i]] = edge_map_vec;
	}
    
    // print result statistics
    std::cout << "Number of samples: " << final_list.size() << std::endl;

    // Random walk on the graph
    int iteration = 1000000;
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // stat
    int max_weight = 0;

    graph g(final_list, edge_weight_map, neighbor_map);
    

    g.randomWalkParallel(iteration);
    g.draw_displacement();

    end = clock();
    std::cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;

    return 0;
}
