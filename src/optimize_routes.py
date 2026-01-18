import numpy as np
import heapq
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

class Location:
    def __init__(self, id, name, lat, lon, time_window=(0, 1440)):
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.time_window = time_window

class RouteOptimizer:
    def __init__(self):
        self.locations = []
        self.distance_matrix = None
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        earth_radius = 6371
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return earth_radius * c
    
    def create_distance_matrix(self, locs):
        self.locations = locs
        n = len(locs)
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = self.haversine_distance(
                        locs[i].lat, locs[i].lon,
                        locs[j].lat, locs[j].lon
                    )
                    self.distance_matrix[i][j] = d
        return self.distance_matrix
    
    def dijkstra(self, start, end):
        n = len(self.distance_matrix)
        dist = [float('inf')] * n
        dist[start] = 0
        prev = [-1] * n
        visited = set()
        pq = [(0, start)]
        while pq:
            d, curr = heapq.heappop(pq)
            if curr in visited:
                continue
            visited.add(curr)
            if curr == end:
                break
            for next_node in range(n):
                if next_node not in visited and self.distance_matrix[curr][next_node] > 0:
                    new_dist = d + self.distance_matrix[curr][next_node]
                    if new_dist < dist[next_node]:
                        dist[next_node] = new_dist
                        prev[next_node] = curr
                        heapq.heappush(pq, (new_dist, next_node))
        path = []
        node = end
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()
        return dist[end], path
    
    def heuristic(self, idx1, idx2):
        return self.haversine_distance(
            self.locations[idx1].lat, self.locations[idx1].lon,
            self.locations[idx2].lat, self.locations[idx2].lon
        )
    def astar(self, start, end):
        n = len(self.distance_matrix)
        g = [float('inf')] * n
        g[start] = 0
        f = [float('inf')] * n
        f[start] = self.heuristic(start, end)
        prev = [-1] * n
        open_set = [(f[start], start)]
        closed = set()
        while open_set:
            _, curr = heapq.heappop(open_set)
            if curr in closed:
                continue
            if curr == end:
                break
            closed.add(curr)
            for next_node in range(n):
                if next_node in closed or self.distance_matrix[curr][next_node] == 0:
                    continue
                tent_g = g[curr] + self.distance_matrix[curr][next_node]
                if tent_g < g[next_node]:
                    prev[next_node] = curr
                    g[next_node] = tent_g
                    f[next_node] = tent_g + self.heuristic(next_node, end)
                    heapq.heappush(open_set, (f[next_node], next_node))
        path = []
        node = end
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()
        return g[end], path 
    
    def ortools_vrp(self, depot_idx=0, num_vehicles=1, time_windows=None):
        distance_matrix_m = (self.distance_matrix * 1000).astype(int).tolist()
        manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix_m), 
            num_vehicles, 
            depot_idx
        )
        routing = pywrapcp.RoutingModel(manager)
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix_m[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        if time_windows:
            def time_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return time_windows[from_node][0]
            time_callback_index = routing.RegisterUnaryTransitCallback(time_callback)
            routing.AddDimension(
                time_callback_index,
                30,
                1440,
                False,
                'Time'
            )
            time_dimension = routing.GetDimensionOrDie('Time')
            for location_idx, (start, end) in enumerate(time_windows):
                index = manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(start, end)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._extract_ortools_solution(manager, routing, solution, num_vehicles)
        else:
            return None
    
    def _extract_ortools_solution(self, manager, routing, solution, num_vehicles):
        routes = []
        total_distance = 0
        for vehicle_id in range(num_vehicles):
            route = []
            route_distance = 0
            index = routing.Start(vehicle_id)
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            route.append(manager.IndexToNode(index))
            routes.append({
                'vehicle_id': vehicle_id,
                'route': route,
                'distance_m': route_distance,
                'distance_km': route_distance / 1000
            })
            total_distance += route_distance
        
        return {
            'routes': routes,
            'total_distance_km': total_distance / 1000,
            'num_vehicles': num_vehicles
        }
    
    def optimize_route(self, locations, algorithm='ortools', **kwargs):
        self.create_distance_matrix(locations)
        
        if algorithm == 'dijkstra':
            start_idx = kwargs.get('start_idx', 0)
            end_idx = kwargs.get('end_idx', len(locations) - 1)
            return self.dijkstra(start_idx, end_idx)
        
        elif algorithm == 'astar':
            start_idx = kwargs.get('start_idx', 0)
            end_idx = kwargs.get('end_idx', len(locations) - 1)
            return self.astar(start_idx, end_idx)
        
        elif algorithm == 'ortools':
            depot_idx = kwargs.get('depot_idx', 0)
            num_vehicles = kwargs.get('num_vehicles', 1)
            time_windows = kwargs.get('time_windows', None)
            return self.ortools_vrp(depot_idx, num_vehicles, time_windows)
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

def demo_route_optimization():
    locations = [
        Location(0, "Warehouse", 40.7128, -74.0060),
        Location(1, "Customer A", 40.7580, -73.9855),
        Location(2, "Customer B", 40.7489, -73.9680),
        Location(3, "Customer C", 40.7614, -73.9776),
        Location(4, "Customer D", 40.7829, -73.9654),
    ]
    optimizer = RouteOptimizer()
    print("Route Optimization Demo")
    print("-" * 50)
    print("\nDijkstra's Algorithm (Warehouse to Customer D)")
    distance, path = optimizer.optimize_route(
        locations, 
        algorithm='dijkstra',
        start_idx=0,
        end_idx=4
    )
    print(f"Distance: {distance:.2f} km")
    print(f"Path: {' -> '.join([locations[i].name for i in path])}")
    print("\nA* Algorithm (Warehouse to Customer D)")
    distance, path = optimizer.optimize_route(
        locations, 
        algorithm='astar',
        start_idx=0,
        end_idx=4
    )
    print(f"Distance: {distance:.2f} km")
    print(f"Path: {' -> '.join([locations[i].name for i in path])}")
    print("\nGoogle OR-Tools VRP (All Customers)")
    result = optimizer.optimize_route(
        locations, 
        algorithm='ortools',
        depot_idx=0,
        num_vehicles=1
    )
    if result:
        print(f"Total Distance: {result['total_distance_km']:.2f} km")
        for route_info in result['routes']:
            route = route_info['route']
            print(f"Vehicle {route_info['vehicle_id']}: "
                  f"{' -> '.join([locations[i].name for i in route])}")
            print(f"Distance: {route_info['distance_km']:.2f} km")

if __name__ == "__main__":
    demo_route_optimization()
