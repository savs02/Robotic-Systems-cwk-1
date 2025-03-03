from queue import PriorityQueue
from .occupancy_grid import OccupancyGrid
from .planner_base import PlannerBase
from .search_grid import SearchGridCell 

class DijkstraPlanner(PlannerBase):

    def __init__(self, occupancy_grid: OccupancyGrid):
        PlannerBase.__init__(self, occupancy_grid)
        self.priority_queue = PriorityQueue()  

    def push_cell_onto_queue(self, cell: SearchGridCell):
        self.priority_queue.put((cell.path_cost, cell))

    def is_queue_empty(self) -> bool:
        return self.priority_queue.empty()

    def pop_cell_from_queue(self) -> SearchGridCell:
        _, cell = self.priority_queue.get()
        return cell

    def resolve_duplicate(self, cell: SearchGridCell, parent_cell: SearchGridCell):
        new_cost = parent_cell.path_cost + 1
        if new_cost < cell.path_cost:
            cell.path_cost = new_cost
            cell.set_parent(parent_cell)
            self.push_cell_onto_queue(cell)
