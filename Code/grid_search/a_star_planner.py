'''
Created on 2 Jan 2022

@author: ucacsjj
'''

import math
import heapq

from .dijkstra_planner import DijkstraPlanner
from .occupancy_grid import OccupancyGrid

class AStarPlanner(DijkstraPlanner):
    def __init__(self, occupancy_grid: OccupancyGrid):
        DijkstraPlanner.__init__(self, occupancy_grid)
        
    def heuristic(self, node, goal):  # Euclidean heuristic
        x1, y1 = node 
        x2, y2 = goal
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Q2d:
    # Complete implementation of A*
    def plan(self, start, goal): 
        open_set = []  # Priority queue
        heapq.heappush(open_set, (0, start))

        g_values = {start: 0}  # Cost from start to each node
        f_values = {start: self.heuristic(start, goal)}  # estimated cost
        came_from = {} 

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.occupancy_grid.get_neighbors(current):
                tentative_g = g_values[current] + self.occupancy_grid.get_transition_cost(current, neighbor)

                if neighbor not in g_values or tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    f_values[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_values[neighbor], neighbor))
                    came_from[neighbor] = current

        return None  # No path found

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]
