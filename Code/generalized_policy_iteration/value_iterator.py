'''
Created on 29 Jan 2022

@author: ucacsjj
'''

from .dynamic_programming_base import DynamicProgrammingBase

# This class ipmlements the value iteration algorithm

class ValueIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the value iteration
        # algorithm is carried out is carried out.
        self._max_optimal_value_function_iterations = 2000
   
    # Method to change the maximum number of iterations
    def set_max_optimal_value_function_iterations(self, max_optimal_value_function_iterations):
        self._max_optimal_value_function_iterations = max_optimal_value_function_iterations

    #    
    def solve_policy(self):

        # Initialize the drawers
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        self._compute_optimal_value_function()
 
        self._extract_policy()
        
        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()
        
        return self._v, self._pi

    # Q3f:
    # Finish the implementation of the methods below.

    def _compute_optimal_value_function(self):

        # This method returns no value.
        # The method updates self._v

        # Get environment and map
        environment = self._environment
        map_ = environment.map()

        # Loop until the maximum number of iterations
        for _ in range(self._max_optimal_value_function_iterations):

            # Track the maximum change in the value function
            delta = 0

            # Sweep systematically over all the states
            for x in range(map_.width()):
                for y in range(map_.height()):
                    
                    # We skip obstructions and terminals. If a cell is obstructed,
                    # there's no action the robot can take to access it, so it doesn't
                    # count. If the cell is terminal, it executes the terminal action
                    # state. The value of the terminal cell is the reward.
                    if map_.cell(x, y).is_obstruction() or map_.cell(x, y).is_terminal():
                        continue

                    old_v = self._v.value(x, y)

                    # Compute the best Q-value among all possible actions
                    best_q = float('-inf')

                    # We assume environment.available_actions() returns
                    # a structure from which we can iterate over all actions:
                    # e.g. for action in range(environment.available_actions().n):
                    # (Adjust if your environment provides them differently)
                    for action in range(environment.available_actions().n):
                        s_prime, r, p = environment.next_state_and_reward_distribution((x, y), action)

                        # Compute the Q-value
                        q_value = 0
                        for i in range(len(p)):
                            s_prime_coords = s_prime[i].coords()
                            q_value += p[i] * (r[i] + self._gamma * self._v.value(s_prime_coords[0], s_prime_coords[1]))

                        if q_value > best_q:
                            best_q = q_value

                    # Update the value function
                    self._v.set_value(x, y, best_q)

                    # Update the maximum deviation
                    delta = max(delta, abs(old_v - best_q))

            # Terminate if the maximum change is small
            if delta < self._theta:
                break

    def _extract_policy(self):

        # This method returns no value.
        # The policy is in self._pi

        # Get environment and map
        environment = self._environment
        map_ = environment.map()

        # For each non-obstructed and non-terminal cell, pick the action
        # that maximizes the Q-value given the final value function
        for x in range(map_.width()):
            for y in range(map_.height()):
                
                if map_.cell(x, y).is_obstruction() or map_.cell(x, y).is_terminal():
                    continue

                best_action = None
                best_q = float('-inf')

                for action in range(environment.available_actions().n):
                    s_prime, r, p = environment.next_state_and_reward_distribution((x, y), action)

                    q_value = 0
                    for i in range(len(p)):
                        s_prime_coords = s_prime[i].coords()
                        q_value += p[i] * (r[i] + self._gamma * self._v.value(s_prime_coords[0], s_prime_coords[1]))

                    if q_value > best_q:
                        best_q = q_value
                        best_action = action

                # Set the best action for this state in the policy
                self._pi.set_action(x, y, best_action)
