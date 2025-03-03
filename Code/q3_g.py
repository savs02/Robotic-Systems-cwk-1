#!/usr/bin/env python3

'''
Created on 3 Feb 2022

@author: ucacsjj
'''

import time
from common.scenarios import *
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from p2.low_level_environment import LowLevelEnvironment
from p2.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    
    # Get the map for the scenario
    #airport_map, drawer_height = three_row_scenario()
    airport_map, drawer_height = full_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = LowLevelEnvironment(airport_map)
    
    # Use a variable for p
    p = 0.6
    airport_environment.set_nominal_direction_probability(p)
    
    # Create the policy iterator
    policy_solver = PolicyIterator(airport_environment)
    
    # --- Monkey-patch PolicyIterator to count iterations ---
    policy_solver.iteration_counter = 0
    original_improve_policy = PolicyIterator._improve_policy
    def patched_improve_policy(self):
        self.iteration_counter += 1
        return original_improve_policy(self)
    policy_solver._improve_policy = patched_improve_policy.__get__(policy_solver, PolicyIterator)
    # ---------------------------------------------------------
    
    # Set up initial state
    policy_solver.initialize()
        
    # Bind the drawer with the solver
    policy_drawer = LowLevelPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)
    
    value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_function_drawer)
    
    # Time the execution of Policy Iteration
    start_time = time.time()
    v, pi = policy_solver.solve_policy()
    policy_duration = time.time() - start_time
    
    # Save screenshots
    policy_drawer.save_screenshot(f"policy_iteration_policy_{p}.pdf")
    value_function_drawer.save_screenshot(f"policy_iteration_value_function_{p}.pdf")
    
    print("\nPolicy Iteration converged after", policy_solver.iteration_counter, "iterations in", f"{policy_duration:.3f}", "seconds.")
    
    # Wait for a key press
    value_function_drawer.wait_for_key_press()
    
    
    # Q3g:
    from generalized_policy_iteration.value_iterator import ValueIterator
    
    # Create the value iterator instance
    value_solver = ValueIterator(airport_environment)
    
    # Monkey-patch ValueIterator to count iterations
    value_solver.iteration_counter = 0
    original_compute_optimal = ValueIterator._compute_optimal_value_function
    def patched_compute_optimal(self):
        environment = self._environment
        map_ = environment.map()
        for _ in range(self._max_optimal_value_function_iterations):
            self.iteration_counter += 1
            delta = 0
            for x in range(map_.width()):
                for y in range(map_.height()):
                    if map_.cell(x, y).is_obstruction() or map_.cell(x, y).is_terminal():
                        continue
                    old_v = self._v.value(x, y)
                    best_q = float('-inf')
                    for action in range(environment.available_actions().n):
                        s_prime, r, p = environment.next_state_and_reward_distribution((x, y), action)
                        q_value = 0
                        for t in range(len(p)):
                            s_prime_coords = s_prime[t].coords()
                            q_value += p[t] * (r[t] + self._gamma * self._v.value(s_prime_coords[0], s_prime_coords[1]))
                        if q_value > best_q:
                            best_q = q_value
                    self._v.set_value(x, y, best_q)
                    delta = max(delta, abs(old_v - best_q))
            if delta < self._theta:
                break
    value_solver._compute_optimal_value_function = patched_compute_optimal.__get__(value_solver, ValueIterator)
    
    # Set up initial state for value iteration
    value_solver.initialize()
    
    # Bind the drawers for value iteration
    value_policy_drawer = LowLevelPolicyDrawer(value_solver.policy(), drawer_height)
    value_solver.set_policy_drawer(value_policy_drawer)
    
    value_function_drawer_vi = ValueFunctionDrawer(value_solver.value_function(), drawer_height)
    value_solver.set_value_function_drawer(value_function_drawer_vi)
    
    # Time the execution of Value Iteration
    start_time = time.time()
    v_val, pi_val = value_solver.solve_policy()
    value_duration = time.time() - start_time
    
    # Save screenshots
    value_policy_drawer.save_screenshot(f"value_iteration_policy_{p}.pdf")
    value_function_drawer_vi.save_screenshot(f"value_iteration_value_function_{p}.pdf")
    
    print("Value Iteration converged after", value_solver.iteration_counter, "iterations in", f"{value_duration:.3f}", "seconds.")
    
    # Wait for a key press so you can compare the two results
    value_function_drawer_vi.wait_for_key_press()
    
    # ----------------------------
    # Discussion:
    # ----------------------------
    # The printed iteration counts and the execution durations show the number of outer iterations
    # (or value sweeps) needed for each algorithm and their total run time.
    # Typically, Policy Iteration converges in fewer outer iterations but may have a higher per-iteration cost
    # due to the iterative policy evaluation, while Value Iteration may require more sweeps.
    # These metrics, along with the final policy quality, help determine which algorithm is more suitable.
    
    input("Press any key to exit.")
