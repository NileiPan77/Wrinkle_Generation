#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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

#define GRID_SIZE 1024
#define N 20

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


std::vector<std::pair<int, int>> poissonSample() {
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
    std::vector<std::pair<int, int>> active_list;
    active_list.push_back(std::make_pair(i, j));

    // initialize the final sample list
    std::vector<std::pair<int, int>> final_list;

    // insert the first sample into the final list
    final_list.push_back(std::make_pair(i, j));

    // insert the first sample into the active list
    active_list.push_back(std::make_pair(i, j));

    // initialize max number of attempts
    int max_attempts = 30;

    // initialize the radius
    float radius = (N - 1) / 2;

    // generate the samples
    while (!active_list.empty()) {
        // randomly select a sample from the active list
        int index = floor(dis(gen) * active_list.size());
        std::pair<int, int> sample = active_list[index];

        // initialize the flag
        bool flag = false;

        // Try up to max_attempts times to find a candidate
        for (size_t i = 0; i < max_attempts; i++) {
            // generate a random angle
            float angle = dis(gen) * 2 * M_PI;

            // generate a random radius
            float r = dis(gen) * radius + radius;

            // calculate the candidate
            int x = floor(sample.first + r * cos(angle));
            int y = floor(sample.second + r * sin(angle));

            // check if the candidate is valid
            if (is_validPoint(x, y, GRID_SIZE, grid)) {
                // insert the candidate into the active list
                active_list.push_back(std::make_pair(x, y));

                // insert the candidate into the final list
                final_list.push_back(std::make_pair(x, y));

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

bool compare_coord(std::vector<double> p1, std::vector<double> p2) {
	return p1[0] < p2[0] ? true : p1[0] == p2[0] && p1[1] < p2[1];
}

int main(int argc, char* argv[]) {
    time_t start, end;
    start = clock();

    std::cout << "Poisson Disk Sampling" << std::endl;

    std::vector<std::pair<int, int>> final_list = poissonSample();
    

    // Use kdtree to find k-nearest neighbors for edge connection
    Kdtree::KdNodeVector nodes;
    for (int i = 0; i < final_list.size(); i++) {
        std::vector<double> point(2);
        point[0] = final_list[i].first;
        point[1] = final_list[i].second;
		nodes.push_back(Kdtree::KdNode(point));
	}

    // build the kdtree
    Kdtree::KdTree tree(&nodes);
    
    // k nearest neighbors
    int k = 4;

    // edge map
    std::unordered_map<std::pair<int, int>, std::vector<Kdtree::KdNodeVector>, pair_hash> edge_map;

    // edge weight map
    std::unordered_map<std::pair<std::pair<int,int>, std::pair<int, int>>, double, pairOfpairHash> edge_weight_map;

    // edge map as a vector of vec2
    std::vector<vec2> edge_map_vec;
    // connect the edges
    for (int i = 0; i < final_list.size(); i++) {
        Kdtree::KdNodeVector result;
        std::vector<double> point(2);
        point[0] = final_list[i].first;
        point[1] = final_list[i].second;
        tree.k_nearest_neighbors(point, k, &result);
        // edge_map.insert(std::make_pair(final_list[i], result));

        // Remove the first element in the result (which is the point itself)
        result.erase(result.begin());

        edge_map[final_list[i]].push_back(result);
        for (int j = 0; j < result.size(); j++) {
            edge_map_vec.push_back(std::make_pair(result[j].point[0], result[j].point[1]));
            edge_weight_map[std::make_pair(final_list[i], std::make_pair(result[j].point[0], result[j].point[1]))] = 0;
		}
	}

    // print result statistics
    std::cout << "Number of samples: " << final_list.size() << std::endl;

    // Random walk on the graph
    int iteration = 100000;
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // stat
    int max_weight = 0;

    for (int i = 0; i < iteration; i++) {
		// Randomly select a point
		int index = floor(dis(gen) * final_list.size());
		std::pair<int, int> current_point = final_list[index];

		// Randomly select a neighbor
		int neighbor_index = floor(dis(gen) * edge_map[current_point][0].size());
		std::pair<int, int> neighbor_point(edge_map[current_point][0][neighbor_index].point[0], edge_map[current_point][0][neighbor_index].point[1]);

		// Update the weight
		edge_weight_map[std::make_pair(current_point, neighbor_point)] += 1;

        // Update the max weight
        if (edge_weight_map[std::make_pair(current_point, neighbor_point)] > max_weight) {
			max_weight = edge_weight_map[std::make_pair(current_point, neighbor_point)];
		}
	}

    // Normalize the weight
    for (auto it = edge_weight_map.begin(); it != edge_weight_map.end(); it++) {
        it->second /= max_weight;
    }


    // Draw the final list
    cv::Mat image(GRID_SIZE, GRID_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int i = 0; i < final_list.size(); i++) {
        image.at<cv::Vec3b>(final_list[i].first, final_list[i].second)[0] = 0;
        image.at<cv::Vec3b>(final_list[i].first, final_list[i].second)[1] = 0;
        image.at<cv::Vec3b>(final_list[i].first, final_list[i].second)[2] = 0;
    }

    // Draw edges
    for (int i = 0; i < final_list.size(); i++) {
        cv::Point current_point(final_list[i].second, final_list[i].first);
        for (int j = 0; j < edge_map[final_list[i]].size(); j++) {
            for (int k = 0; k < edge_map[final_list[i]][j].size(); k++) {
				cv::Point neighbor_point(edge_map[final_list[i]][j][k].point[1], edge_map[final_list[i]][j][k].point[0]);
				cv::line(image, current_point, neighbor_point, cv::Scalar(255, 255, 255) - cv::Scalar(255, 255, 255) * edge_weight_map[std::make_pair(final_list[i], std::make_pair(edge_map[final_list[i]][j][k].point[0], edge_map[final_list[i]][j][k].point[1]))], 1, 8);
			}
		}
	}
    /*for (auto edge : edges) {
        cv::Point current_point(edge.first[1], edge.first[0]);
        cv::Point neighbor_point(edge.second[1], edge.second[0]);
        cv::line(image, current_point, neighbor_point, cv::Scalar(0, 0, 255), 1, 8);
    }*/

    // show the image
    cv::imshow("Poisson Disk Sampling", image);

    end = clock();
    std::cout << "Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::waitKey(0);

    return 0;
}
