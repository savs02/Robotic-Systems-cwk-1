#!/usr/bin/env python3
'''
Created on 3 Feb 2022

Experiment to investigate how the policy evaluation parameters affect
the convergence speed (time and iteration count). We run two experiments:
1. Holding theta constant and varying max policy evaluation steps.
2. Holding max policy evaluation steps constant and varying theta.

We count outer policy iterations via monkey-patching without modifying the original class.
@author: ucacsjj
'''

import time

from common.scenarios import full_scenario
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    
    # Set up the map and environment.
    airport_map, drawer_height = full_scenario()
    airport_environment = LowLevelEnvironment(airport_map)
    airport_environment.set_nominal_direction_probability(0.8)  # p = 0.8

    # Experiment 1: Varying max policy evaluation steps (theta held constant).
    print("Experiment 1: Varying max policy evaluation steps (theta held constant).")
    # The default theta in dynamic_programming_base.py is 1e-6
    const_theta = 1e-6
    eval_steps_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    results_eval_steps = []  # Will store tuples: (eval_steps, theta, elapsed_time, iterations)

    for eval_steps in eval_steps_list:
        print(f"\nRunning with max policy evaluation steps = {eval_steps} and theta = {const_theta}")
        
        policy_solver = PolicyIterator(airport_environment)
        policy_solver.set_max_policy_evaluation_steps_per_iteration(eval_steps)
        policy_solver.set_theta(const_theta)
        
        # Monkey-patch _improve_policy to count outer iterations.
        # Get the unbound original method from the class.
        original_improve_policy = PolicyIterator._improve_policy  
        policy_solver.iteration_counter = 0
        
        def patched_improve_policy(self):
            self.iteration_counter += 1
            return original_improve_policy(self)
        
        # Bind the patched method to our instance.
        policy_solver._improve_policy = patched_improve_policy.__get__(policy_solver, PolicyIterator)
        
        policy_solver.initialize()
        
        policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
        policy_solver.set_policy_drawer(policy_drawer)
        value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
        policy_solver.set_value_function_drawer(value_function_drawer)
        
        start_time = time.time()
        v, pi = policy_solver.solve_policy()
        elapsed_time = time.time() - start_time
        
        iterations = policy_solver.iteration_counter
        print(f"Time: {elapsed_time:.3f} s, Policy Iterations: {iterations}")
        
        filename = f"policy_results_evalsteps_{eval_steps}_theta_{const_theta}.jpg"
        policy_drawer.save_screenshot(filename)
        
        results_eval_steps.append((eval_steps, const_theta, elapsed_time, iterations))
    
    # Vary theta, hold max eval steps constant.
    print("\nExperiment 2: Varying theta (max policy evaluation steps held constant).")
    # The default max eval steps per iteration in policy_iterator.py is 100
    const_eval_steps = 100
    theta_list = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000]
    results_theta = []  # Will store tuples: (eval_steps, theta, elapsed_time, iterations)

    for theta in theta_list:
        print(f"\nRunning with max policy evaluation steps = {const_eval_steps} and theta = {theta}")
        
        policy_solver = PolicyIterator(airport_environment)
        policy_solver.set_max_policy_evaluation_steps_per_iteration(const_eval_steps)
        policy_solver.set_theta(theta)
        
        # Monkey-patch _improve_policy to count outer iterations.
        original_improve_policy = PolicyIterator._improve_policy  
        policy_solver.iteration_counter = 0
        
        def patched_improve_policy(self):
            self.iteration_counter += 1
            return original_improve_policy(self)
        
        policy_solver._improve_policy = patched_improve_policy.__get__(policy_solver, PolicyIterator)
        
        policy_solver.initialize()
        
        policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
        policy_solver.set_policy_drawer(policy_drawer)
        value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
        policy_solver.set_value_function_drawer(value_function_drawer)
        
        start_time = time.time()
        v, pi = policy_solver.solve_policy()
        elapsed_time = time.time() - start_time
        
        iterations = policy_solver.iteration_counter
        
        results_theta.append((const_eval_steps, theta, elapsed_time, iterations))
    
    # Print results summary.
    print("\nSummary of Experiment 1 (varying max eval steps, theta fixed):")
    for eval_steps, theta, elapsed, iterations in results_eval_steps:
        print(f"Max Eval Steps: {eval_steps:3}, Theta: {theta:1.0e}, Time: {elapsed:.3f} s, Iterations: {iterations}")
    
    print("\nSummary of Experiment 2 (varying theta, max eval steps fixed):")
    for eval_steps, theta, elapsed, iterations in results_theta:
        print(f"Max Eval Steps: {eval_steps:3}, Theta: {theta:1.0e}, Time: {elapsed:.3f} s, Iterations: {iterations}")
    
    input("Press any key to exit.")
