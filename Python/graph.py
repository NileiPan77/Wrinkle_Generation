import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial import Delaunay, distance
from scipy.stats.qmc import PoissonDisk
from scipy.signal import convolve2d
import pyexr
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import qmc
import multiprocessing as mp
import concurrent.futures
import copy
from collections import defaultdict
import cProfile

def radial_gradient(center, sigma, size=5):
    y, x = np.ogrid[-center[1]:size-center[1], -center[0]:size-center[0]]
    g = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    return g

class Skin:
    def __init__(self, num_pores=100, width=10, height=10, max_depth=100, k = 3, avg_node_distance=0.5, angle=0, \
            a_s=0, a_dist=5, a_cont=0, a_cross=3, a_deposit = 1.0, a_decay = 0.5, a_sim = 1, \
                iterations = 10000, delta_t = 0.03, \
                    cspread = 0.3):
        self.graph = nx.Graph()
        self.width = width
        self.height = height
        self.num_pores = num_pores
        self.max_depth = max_depth
        self.k = k
        self.avg_node_distance = avg_node_distance / max(self.height, self.width)
        self.a_dist = a_dist
        self.a_cont = a_cont
        self.a_cross = a_cross
        angle = np.deg2rad(angle)  # convert angle to radians
        self.angle = angle
        self.a_s = a_s
        self.a_deposit = a_deposit
        self.a_decay = a_decay
        self.a_sim = a_sim
        self.delta_t = delta_t
        self.cspread = cspread
        self.iterations = iterations
        self.preferred_direction = np.array([np.cos(angle), np.sin(angle)])
        self.intersections = defaultdict(set)

        self.create_pores()
        self.create_edges()
        # self.carve_wrinkles()
        self.parallel_carve_wrinkles(mp.cpu_count())
        print('intersection count:', len(self.intersections))
        #self.draw_graph()
        self.draw_displacement_graph()

    def create_pores(self):
        area = self.width * self.height
        num_pores = int(area / self.avg_node_distance ** 2)
        radius = self.avg_node_distance * (2**0.5)

        sampler = PoissonDisk(d=2, radius=radius, hypersphere='surface', ncandidates=30)
        self.pores = sampler.random(n=num_pores) * [self.width, self.height]
        for i, pore in enumerate(self.pores):
            self.graph.add_node(i, position=pore)

    def is_intersection(self, h1, h2):
        # check if two half-edges intersect
        p1, p2 = self.graph.nodes[h1[0]]['position'], self.graph.nodes[h1[1]]['position']
        p3, p4 = self.graph.nodes[h2[0]]['position'], self.graph.nodes[h2[1]]['position']
        denominator = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
        return denominator != 0

    def make_intersection_dict(self, h):
        # make a dictionary of intersections
        h_pos = (self.graph.nodes[h[0]]['position'], self.graph.nodes[h[1]]['position'])
        # print("h_pos: ", h_pos)
        for node in self.graph.nodes:
            if node == h[0]:
                continue
            for edge in self.graph.edges(node):
                edge_pos = (self.graph.nodes[edge[0]]['position'], self.graph.nodes[edge[1]]['position'])
                if self.is_intersection(h_pos, edge_pos):
                    self.intersections[h].add(edge)
            
    def create_edges(self):
        # find k nearest neighbors
        tree = KDTree(self.pores)
        for i, pore in enumerate(self.pores):
            dist, ind = tree.query(pore, k=self.k+1)
            for j in ind[1:]:
                self.graph.add_edge(i, j)
                self.graph[i][j]['weight'] = 0.0
                self.check_intersection(i, j)
        
    def check_intersection(self, node1, node2):
        # get neighbors and neighbors of neighbors of node1 and node2
        neighbors_of_neighbors1 = set()
        for neighbor in set(self.graph.neighbors(node1)):
            neighbors_of_neighbors1.update(self.graph.neighbors(neighbor))
        neighbors_of_neighbors2 = set()
        for neighbor in set(self.graph.neighbors(node2)):
            neighbors_of_neighbors2.update(self.graph.neighbors(neighbor))

        # get edges of neighbors of neighbors of node1 and node2
        edges1 = set()
        for neighbor in neighbors_of_neighbors1:
            edges1.update(self.graph.edges(neighbor))
        edges2 = set()
        for neighbor in neighbors_of_neighbors2:
            edges2.update(self.graph.edges(neighbor))

        # check if any edges intersect with (node1, node2)
        my_edge = tuple(sorted((node1, node2)))
        for edge in edges1:
            if edge == my_edge:
                continue
            if self.is_intersection(my_edge, edge):
                self.intersections[my_edge].add(edge)
        for edge in edges2:
            if edge == my_edge:
                continue
            if self.is_intersection(my_edge, edge):
                self.intersections[my_edge].add(edge)

    
    def orient_prob(self, ang1, alpha, num_pis):
        return min(np.abs(ang1 - alpha), np.abs(ang1 - alpha + num_pis), np.abs(ang1 - alpha - num_pis))

    def p_pref(self, alpha, s, ang ,theta):
        return np.clip(s + np.exp(- self.orient_prob(ang, alpha, np.pi) ** 2 / (theta ** 2)), 0, 1)

    def compute_p_pref(self, h):
        ang = self.angle_between(self.preferred_direction, h[1] - h[0])
        return self.p_pref(self.angle, self.a_s, ang, 0.3)
        
    def unit_vector(self, v):
        return v / np.linalg.norm(v)

    def angle_between(self, v1, v2):
        angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
        return angle
    
    def worker(self, graph, current_node, iterations):
        edge_list = []
        for _ in range(iterations):
            neighbors = list(graph.neighbors(current_node))
            # neighbors_position = [np.array(graph.nodes[neighbor]['position']) for neighbor in neighbors]
            half_nodes = [(current_node, neighbor) for neighbor in neighbors]
            probabilities = [self.p(h)+ 1e-8 for h in half_nodes]
            # print(probabilities)
            probabilities = probabilities / np.sum(probabilities)
            next_node = np.random.choice(neighbors, p=probabilities)
            # next_node = neighbors[np.argmax(probabilities)]
            weight = graph[current_node][next_node]['weight']
            edge_list.append((current_node, next_node))
            weight = weight + self.delta_t * self.a_deposit * (1 - weight)
            graph[current_node][next_node]['weight'] = weight
            current_node = np.random.choice(list(graph.nodes()))
        # apply global decay
        for edge in edge_list:
            graph[edge[0]][edge[1]]['weight'] = graph[edge[0]][edge[1]]['weight'] * (1 - self.delta_t * self.a_decay * (iterations / len(edge_list)))
        return graph

    def parallel_carve_wrinkles(self, num_processes):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Make a list of all the "workers" (i.e., parallel tasks)
            workers = [executor.submit(self.worker, copy.deepcopy(self.graph), \
                np.random.choice(list(self.graph.nodes())), self.iterations) for _ in range(num_processes)]
            
            # Gather all the results and merge them
            for future in concurrent.futures.as_completed(workers):
                result_graph = future.result()
                self.merge_graphs(result_graph)
    
    def merge_graphs(self, other_graph):
        # This function takes another graph and merges its weights into the main graph
        for u, v, data in other_graph.edges(data=True):
            if self.graph.has_edge(u, v):
                self.graph[u][v]['weight'] = self.graph[u][v]['weight'] + data['weight']
            else:
                self.graph.add_edge(u, v, weight=data['weight'])



    def distance(self, node1, node2):
        pos1 = self.graph.nodes[node1]['position']
        pos2 = self.graph.nodes[node2]['position']
        return np.sqrt(np.sum((pos1 - pos2) ** 2.0))

    def gaussian(self, x, sig):
        return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))

    def compute_p_cont(self, h):
        # Implement logic for computing p_cont
        node1, node2 = h
        node1_pos, node2_pos = self.graph.nodes[node1]['position'], self.graph.nodes[node2]['position']
        ang = np.arctan2(node2_pos[1] - node1_pos[1], node2_pos[0] - node1_pos[0])
        p_cont = 0
        for neighbor in self.graph.neighbors(node1):
            edge = (node1, neighbor)
            neighbor_pos = self.graph.nodes[neighbor]['position']
            ang2 = np.arctan2(neighbor_pos[1] - node1_pos[1], neighbor_pos[0] - node1_pos[0])
            if edge != h:
                p_cont += self.graph[edge[0]][edge[1]]['weight'] * (self.a_cont * self.gaussian(self.orient_prob(ang, (ang2-np.pi), 2 * np.pi), self.cspread))\
                    - self.a_sim * self.gaussian(self.orient_prob(ang, ang2, 2 * np.pi), self.cspread)
        return p_cont

    

    def compute_p_cross(self, h):
        # Implement logic for computing p_cross
        node1, node2 = h
        # print('nodes:', h)
        intersected_edges = self.intersections[tuple(sorted((node1, node2)))]
        weight = np.sum([self.graph[edge[0]][edge[1]]['weight'] for edge in intersected_edges])
        # print('weight:', weight)
        
        return weight

    def compute_distance_factor(self, h):
        d_h = np.sqrt(np.sum((h[1] - h[0]) ** 2.0))
        return 1.0 / (d_h ** self.a_dist)

    def p(self, h):
        h_pos = (self.graph.nodes[h[0]]['position'], self.graph.nodes[h[1]]['position'])
        h_index = h
        h = h_pos
        # Distance-dependent factor
        dist_factor = self.compute_distance_factor(h)

        # Probability factors
        p_cont = self.compute_p_cont(h_index)
        p_pref = self.compute_p_pref(h)
        p_cross = self.compute_p_cross(h_index)

        return max(0.0, dist_factor * (p_cont + (1 - self.a_cont) * p_pref - self.a_cross * p_cross))

    def walk(self, start_node):
        # initialize all edge weights to zero
        weights = {edge: 0 for edge in self.graph.edges}

        # current node to start walk
        current_node = start_node

        # TODO: iterate until a termination condition
        while True:  # replace True with termination condition
            neighbors = list(self.graph.neighbors(current_node))
            if not neighbors:
                break  # break if current node has no neighbors

            probabilities = []
            for neighbor in neighbors:
                h = (current_node, neighbor)  # half-edge from current node to neighbor

                P_h = self.p(h)  # probability of selecting the half-edge
                probabilities.append(P_h)

            # select the neighbor with the highest probability
            selected_neighbor = neighbors[np.argmax(probabilities)]

            # increase the weight of the edge to the selected neighbor
            weights[(current_node, selected_neighbor)] += 1  # increment by 1, modify as needed

            # continue walk from the selected neighbor
            current_node = selected_neighbor

        # assign weights to edges
        nx.set_edge_attributes(self.graph, weights, 'weight')
    def mellowmax(self, x):
        n_hat = 16
        beta = 20
        result = (1/beta)*np.log((1/n_hat)*(n_hat - len(x) + np.sum(np.exp(beta*x))))
        return result
    
    def shape_f(self, x):
        return -(1-2*x/3)**3
    
    def get_enclosing_rectangle(self, p1, p2, r):
        x1, y1 = p1
        x2, y2 = p2

        #Determine the smallest rectangle
        left = min(x1, x2)
        bottom = min(y1, y2)
        right = max(x1, x2)
        top = max(y1, y2)

        #Expand the rectangle by the radius
        left -= r
        bottom -= r
        right += r
        top += r

        # Ensure the coordinates are integers for pixel iteration
        left = max(0, int(left))
        bottom = max(0, int(bottom))
        right = int(right)
        top = int(top)

        return left, bottom, right, top

    def point_to_edge_distance(self, p, a, b):
        # Unpack the points
        (x1, y1), (x2, y2), (x0, y0) = a, b, p
        # Handle the special case where AB is a vertical line
        if x1 == x2:
            return abs(x0 - x1)
        # Calculate the direction vector for AB
        D = np.array([x2 - x1, y2 - y1])
        # Calculate the vector from A to P
        v = np.array([x0 - x1, y0 - y1])

        # Calculate the scalar t
        t = np.dot(v, D) / np.dot(D, D)

        # If t is in [0,1], the closest point is on AB
        if 0 <= t <= 1:
            # Calculate the closest point
            closest = np.array([x1, y1]) + t * D
        elif t < 0:
            # The closest point is A
            closest = np.array([x1, y1])
        else:
            # The closest point is B
            closest = np.array([x2, y2])
        # Return the distance from P to the closest point
        return np.linalg.norm(closest - np.array([x0, y0]))

    def fnv1_hash(self, input):
        ret = -2128831035
        prime = 16777619

        b0 = input & 0xFF
        b1 = (input & 0xFF00) >> 8
        b2 = (input & 0xFF0000) >> 16
        b3 = (input & 0xFF000000) >> 24

        ret *= prime
        ret ^= b0

        ret *= prime
        ret ^= b1

        ret *= prime
        ret ^= b2

        ret *= prime
        ret ^= b3

        return ret

    def gradient_2d(self, vec2):
        x = self.fnv1_hash(vec2[0])
        y = self.fnv1_hash(x + vec2[1])
        return np.sin(np.array([x + y, y + y]).astype(np.float32))
    
    def gradient_2d_controlled(self, vec2, a_fperturb = 2000, a_sperturb = 0.35):
        # Use α_fperturb to adjust the frequency of the noise by scaling the input coordinates
        vec2 = np.array(vec2) * a_fperturb
        vec2 = vec2.astype(int)
        
        # Use the FNV-1 hash function to generate pseudorandom numbers based on the input coordinates
        x = self.fnv1_hash(vec2[0])
        y = self.fnv1_hash(x + vec2[1])
        
        # Use α_sperturb to adjust the amplitude of the noise by scaling the output value
        gradient = np.sin(np.array([x + y, y + y]).astype(np.float32)) * a_sperturb
        
        return gradient

    def gradient_3d(self, vec3):
        x = self.fnv1_hash(vec3[0])
        y = self.fnv1_hash(x + vec3[1])
        z = self.fnv1_hash(y + vec3[2])
        return np.sin(np.array([x + z, z + y, z + z]))

    def draw_displacement_graph(self):
        edge_weights = nx.get_edge_attributes(self.graph, 'weight')
        edge_weights_np = np.array(list(edge_weights.values()))
        edge_weights_norm = (edge_weights_np - np.min(edge_weights_np)) / (np.max(edge_weights_np) - np.min(edge_weights_np))
        edge_weights = {edge: edge_weights_norm[i] for i, edge in enumerate(edge_weights.keys())}

        a_wrinkle_width = 0.4
        dpi = 100
        rows = int(self.height * dpi)
        cols = int(self.width * dpi)
        displacement_map = np.zeros((rows, cols))
        displacement_map_values = [[np.array([]) for _ in range(cols)] for _ in range(rows)]
        r_origin = 1.5

        # Calculate node weights
        node_weights = defaultdict(float)
        a_dmin = 0.6
        a_blend = 1.0
        a_pore_width = 0.5
        a_cushion = 0.8
        for node in self.graph.nodes():
            w_max = 0.0
            w_sum = 0.0
            for edge in self.graph.edges(node):
                edge = tuple(sorted(edge))
                w = edge_weights[edge]
                w_max = max(w_max, w)
                w_sum += w
            node_weights[node] = max(w_max + a_dmin, w_max + a_blend*(w_sum - w_max))

        # calculate the edge displacement
        for edge in self.graph.edges():
            wrinkle_width = a_wrinkle_width * edge_weights[edge]
            # print('wrinkle width: ', wrinkle_width)
            if wrinkle_width < 0.001:
                continue
            r = r_origin * wrinkle_width
            node1, node2 = edge
            node1_pos, node2_pos = self.graph.nodes[node1]['position'] * dpi, self.graph.nodes[node2]['position'] * dpi

            left, bottom, right, top = self.get_enclosing_rectangle(node1_pos, node2_pos, r_origin)
            right = min(right, cols - 1)
            top = min(top, rows - 1)
            left = max(left, 0)
            bottom = max(bottom, 0)
            
            for x in range(left, right + 1):
                for y in range(bottom, top + 1):
                    d = self.point_to_edge_distance((x,y), node1_pos, node2_pos)
                    
                    perlin_noise = self.gradient_2d_controlled(np.array([x, y]))
                    x_perturb = int(perlin_noise[0] + x)
                    x_perturb = min(x_perturb, cols - 1)
                    y_perturb = int(perlin_noise[1] + y)
                    y_perturb = min(y_perturb, rows - 1)
                    # d_perturb = self.point_to_edge_distance((x_perturb, y_perturb), node1_pos, node2_pos)
                    if d < r_origin:
                        displacement_map_values[y_perturb][x_perturb] = np.append(displacement_map_values[y_perturb][x_perturb], self.shape_f(d) * wrinkle_width)
                        displacement_map_values[y][x] = np.append(displacement_map_values[y][x], self.shape_f(d) * wrinkle_width)

        for node in self.graph.nodes():
            node_pos = self.graph.nodes[node]['position'] * dpi
            for edge in self.graph.edges(node):
                node_other = edge[0] if edge[0] != node else edge[1]
                node_other_pos = self.graph.nodes[node_other]['position'] * dpi
                edge_length = np.linalg.norm(self.graph.nodes[edge[0]]['position'] * dpi - self.graph.nodes[edge[1]]['position'] * dpi)
                for i in range(1, int(edge_length)+1):
                    edge_direction = (node_other_pos - node_pos) / edge_length
                    edge_point = node_pos + edge_direction * i
                    dist = np.linalg.norm(edge_point - node_pos)
                    # print('dist: ', dist)
                    x, y = edge_point
                    x = int(x)
                    y = int(y)
                    # print(f'weight at {x}, {y}: ', node_weights[node] * a_cushion * np.exp(-dist))
                    displacement_map_values[y][x] = np.append(displacement_map_values[y][x], -node_weights[node] * np.exp(- a_cushion * dist))

        for x in range(cols):
            for y in range(rows):
                displacement_map[y, x] = self.mellowmax(displacement_map_values[y][x])

        kernel = [[-1, -1, -1, -1, -1],
                    [-1,  2.5,  2.5,  2.5, -1],
                    [-1,  2.5,  4,  2.5, -1],
                    [-1,  2.5,  2.5,  2.5, -1],
                    [-1, -1, -1, -1, -1]]
        kernel = np.array(kernel) / 8
        displacement_map = convolve2d(displacement_map, kernel, mode='same', boundary='symm')
        
        # np.savetxt("displacement_map.csv", displacement_map_values, delimiter=",")
        pyexr.write("displacement_map_4.exr", displacement_map)
        displacement_map = (displacement_map * 255).astype(np.uint8)
        
        plt.figure(figsize=(self.width, self.height), dpi=100)
        plt.imshow(displacement_map, cmap='gray_r')
        plt.show()
        return displacement_map

    def draw_graph(self):
        positions = nx.get_node_attributes(self.graph, 'position')
        

        plt.figure(figsize=(self.width, self.height), dpi=100)
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, positions, node_color='black', node_size=0.01)
        
        
        # Draw edges 
        # nx.draw_networkx_edges(graph, positions, edge_color='lightgreen', width=1)
        edge_colors = [self.graph[u][v]['weight'] for u,v in self.graph.edges()]
        edge_width = ((np.array(edge_colors) + 1.0) ** 2 - 1)
        # print weights
        print(edge_width)
        nx.draw_networkx_edges(self.graph, positions, edge_color=edge_width, edge_cmap=plt.cm.gray_r, width=0.01)

        # no axis
        plt.axis('off')
        plt.margins(0)
        # plt.show()
        # save graph to file
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig('graph.png', pad_inches=0)

def main():
    skin = Skin(k = 5, avg_node_distance=0.15, \
        width=40.96 / 4.0, height=40.96 / 4.0, \
            angle=0, a_s=0.01, \
                a_dist=0, a_cont=0.5, a_cross=0, a_deposit = 1, a_sim = 0, \
                    iterations=10000)

if __name__ == '__main__':
    # Create a skin simulation
    cProfile.run('main()', sort='cumtime')
    # main()