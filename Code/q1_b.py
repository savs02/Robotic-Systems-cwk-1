#!/usr/bin/env python3

'''
Created on 27 Jan 2022

@author: ucacsjj
'''

from common.airport_map_drawer import AirportMapDrawer
from common.scenarios import full_scenario
from p1.high_level_actions import HighLevelActionType
from p1.high_level_environment import HighLevelEnvironment, PlannerType
import numpy as np
import math
import statistics

def calculate_euclidean_distance(point1, point2):
    """Calculate straight-line distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def analyze_planner_performance(planner_type_name, data_dict, all_rubbish_bins):
    """Analyze and print performance metrics for a planner"""
    
    total_path_cost = sum(data[0] for data in data_dict.values())
    total_cells_visited = sum(data[1] for data in data_dict.values())
    
    path_efficiency_ratios = []
    bin_coords = [bin.coords() for bin in all_rubbish_bins]
    bin_coords.insert(0, (0, 0)) 
    
    for i in range(1, len(bin_coords)):
        actual_path_cost = data_dict[i][0]
        straight_line_distance = calculate_euclidean_distance(bin_coords[i-1], bin_coords[i])
        efficiency_ratio = actual_path_cost / straight_line_distance if straight_line_distance > 0 else 0
        path_efficiency_ratios.append(efficiency_ratio)
    
    planning_efficiencies = [data[1] / data[0] if data[0] > 0 else 0 for data in data_dict.values()]
    
    path_cost_stddev = statistics.stdev([data[0] for data in data_dict.values()])
    cells_visited_stddev = statistics.stdev([data[1] for data in data_dict.values()])
    
    total_grid_cells = 2400  
    avg_coverage_percentage = (sum(data[1] for data in data_dict.values()) / len(data_dict)) / total_grid_cells * 100
    

    distance_buckets = {"short": [], "medium": [], "long": []}
    for i in range(1, len(bin_coords)):
        distance = calculate_euclidean_distance(bin_coords[i-1], bin_coords[i])
        if distance < 10:
            distance_buckets["short"].append((i, data_dict[i]))
        elif distance < 20:
            distance_buckets["medium"].append((i, data_dict[i]))
        else:
            distance_buckets["long"].append((i, data_dict[i]))
    

    print(f"\n=== {planner_type_name} Performance Analysis ===")
    print(f"Total Path Cost: {total_path_cost:.2f}")
    print(f"Total Cells Visited: {total_cells_visited}")
    print(f"Average Path Efficiency Ratio: {sum(path_efficiency_ratios)/len(path_efficiency_ratios):.2f}")
    print(f"Average Planning Efficiency (cells/unit path): {sum(planning_efficiencies)/len(planning_efficiencies):.2f}")
    print(f"Path Cost Standard Deviation: {path_cost_stddev:.2f}")
    print(f"Cells Visited Standard Deviation: {cells_visited_stddev:.2f}")
    print(f"Average Exploration Coverage: {avg_coverage_percentage:.2f}%")
    
    # Performance by distance
    print("\nPerformance by Distance:")
    for category, bins in distance_buckets.items():
        if bins:
            avg_cost = sum(bin_data[1][0] for bin_data in bins) / len(bins)
            avg_cells = sum(bin_data[1][1] for bin_data in bins) / len(bins)
            print(f"  {category.capitalize()} distances - Avg Cost: {avg_cost:.2f}, Avg Cells: {avg_cells:.2f}")
    
    return {
        "total_path_cost": total_path_cost,
        "total_cells_visited": total_cells_visited,
        "avg_path_efficiency": sum(path_efficiency_ratios)/len(path_efficiency_ratios),
        "avg_planning_efficiency": sum(planning_efficiencies)/len(planning_efficiencies),
        "path_cost_stddev": path_cost_stddev,
        "cells_visited_stddev": cells_visited_stddev,
        "avg_coverage_percentage": avg_coverage_percentage,
        "by_distance": {
            k: {"count": len(v), 
                "avg_cost": sum(bin_data[1][0] for bin_data in v) / len(v) if v else 0,
                "avg_cells": sum(bin_data[1][1] for bin_data in v) / len(v) if v else 0
               } 
            for k, v in distance_buckets.items()
        }
    }

if __name__ == '__main__':
    airport_map, drawer_height = full_scenario()
    all_rubbish_bins = airport_map.all_rubbish_bins()
    
    results = {}
    
    airport_environment = HighLevelEnvironment(airport_map, PlannerType.BREADTH_FIRST)
    airport_environment.show_graphics(True)
    airport_environment.show_verbose_graphics(False)
    
    action = (HighLevelActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (0, 0))
    observation, reward, done, info = airport_environment.step(action)
    
    bfs_data = {}
    bin_number = 1
    for rubbish_bin in all_rubbish_bins:
        action = (HighLevelActionType.DRIVE_ROBOT_TO_NEW_POSITION, rubbish_bin.coords())
        observation, reward, done, info = airport_environment.step(action)
        screen_shot_name = f'bin_{bin_number:02}.pdf'
        airport_environment.search_grid_drawer().save_screenshot(screen_shot_name)
        bfs_data[bin_number] = [info.path_travel_cost, info.number_of_cells_visited]
        bin_number += 1
    
    airport_environment = HighLevelEnvironment(airport_map, PlannerType.DEPTH_FIRST)
    airport_environment.show_graphics(True)
    airport_environment.show_verbose_graphics(False)
    
    action = (HighLevelActionType.TELEPORT_ROBOT_TO_NEW_POSITION, (0, 0))
    observation, reward, done, info = airport_environment.step(action)
    
    dfs_data = {}
    bin_number = 1
    
    for rubbish_bin in all_rubbish_bins:
        action = (HighLevelActionType.DRIVE_ROBOT_TO_NEW_POSITION, rubbish_bin.coords())
        observation, reward, done, info = airport_environment.step(action)
        screen_shot_name = f'bin_{bin_number:02}.pdf'
        airport_environment.search_grid_drawer().save_screenshot(screen_shot_name)
        dfs_data[bin_number] = [info.path_travel_cost, info.number_of_cells_visited]
        bin_number += 1
    
    bfs_results = analyze_planner_performance("BFS", bfs_data, all_rubbish_bins)
    dfs_results = analyze_planner_performance("DFS", dfs_data, all_rubbish_bins)
    
    print("\n=== Comparison Summary ===")
    print(f"Path Cost Ratio (DFS/BFS): {dfs_results['total_path_cost']/bfs_results['total_path_cost']:.2f}x")
    print(f"Cells Visited Ratio (DFS/BFS): {dfs_results['total_cells_visited']/bfs_results['total_cells_visited']:.2f}x")
    print(f"Path Efficiency Ratio (BFS/DFS): {bfs_results['avg_path_efficiency']/dfs_results['avg_path_efficiency']:.2f}x")
    print(f"Planning Efficiency Ratio (BFS/DFS): {bfs_results['avg_planning_efficiency']/dfs_results['avg_planning_efficiency']:.2f}x")