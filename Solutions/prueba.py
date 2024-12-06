# Bloque 2: Definición del Modelo de Optimización

import numpy as np
import requests
from pyomo.environ import *
from pyomo.environ import SolverFactory
from pyomo.environ import value
from amplpy import modules

class TransportationModel:
    def __init__(self, vehicles, depots, clients):
        self.vehicles = vehicles
        self.depots = depots
        self.clients = clients

        print(f"Cliente 1: {self.clients[0]}")


        self.all_coords = [(depot.longitude, depot.latitude) for depot in depots] + \
                          [(client.longitude, client.latitude) for client in clients]

        self.index_to_location_id = {}
        self.location_id_to_index = {}
        idx = 1

        for depot in depots:
            self.index_to_location_id[idx] = depot.location_id
            self.location_id_to_index[depot.location_id] = idx
            idx += 1
        for client in clients:
            self.index_to_location_id[idx] = client.location_id
            self.location_id_to_index[client.location_id] = idx
            idx += 1

        self.car_matrix_distance = None
        self.car_matrix_time = None
        self.drone_matrix_distance = None
        self.drone_matrix_time = None
        self.model = ConcreteModel()

    def build_distance_matrices(self):
        # Construir cadenas de coordenadas para OSRM
        coords_str = ';'.join([f"{lon},{lat}" for lon, lat in self.all_coords])

        # URL de la API OSRM
        url = f"https://router.project-osrm.org/table/v1/driving/{coords_str}"

        # Parámetros de la solicitud
        params = {
            'sources': ';'.join(map(str, range(len(self.all_coords)))),
            'destinations': ';'.join(map(str, range(len(self.all_coords)))),
            'annotations': 'duration,distance'
        }

        # Enviar la solicitud
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error al conectar con la API OSRM: {e}")
            exit()

        data = response.json()

        # Extraer matrices de distancia y tiempo
        self.car_matrix_distance = np.array(data['distances']) / 1000  # Convertir a km
        self.car_matrix_time = np.array(data['durations']) / 60         # Convertir a minutos

        # Calcular matriz para drones usando fórmula de Haversine
        self.drone_matrix_distance = self.calculate_haversine_matrix()
        self.drone_matrix_time = self.drone_matrix_distance / 40 * 60  # Asumiendo velocidad de 40 km/h

        # Rellenar la diagonal con un valor alto para evitar rutas de un nodo a sí mismo
        np.fill_diagonal(self.car_matrix_distance, 9999999)
        np.fill_diagonal(self.car_matrix_time, 9999999)
        np.fill_diagonal(self.drone_matrix_distance, 9999999)
        np.fill_diagonal(self.drone_matrix_time, 9999999)

    def calculate_haversine_matrix(self):
        def haversine(lon1, lat1, lon2, lat2):
            lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Radio de la Tierra en km
            return c * r

        num_nodes = len(self.all_coords)
        matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    origin = self.all_coords[i]
                    destination = self.all_coords[j]
                    matrix[i][j] = haversine(origin[0], origin[1], destination[0], destination[1])

        return matrix

    def build_model(self):
        model = self.model
        num_depots = len(self.depots)
        num_clients = len(self.clients)
        num_nodes = len(self.all_coords)
        num_cars = len(self.vehicles)

        # Verificar que el número de vehículos coincide
        assert num_cars == len(self.vehicles), "El número de vehículos no coincide con la lista de vehículos."

        # Conjuntos
        model.depots_set = RangeSet(1, num_depots)
        print(f"Depots_set definido: {[i for i in model.depots_set]}")
        model.clients_set = RangeSet(num_depots + 1, num_nodes)
        print(model.clients_set)
        model.nodes_set = RangeSet(1, num_nodes)
        model.cars_set = RangeSet(1, num_cars)

        # Variables de decisión
        model.x = Var(model.nodes_set, model.nodes_set, model.cars_set, domain=Binary)
        model.u = Var(model.clients_set, model.cars_set, domain=NonNegativeReals, bounds=(0, num_clients))

        # Función Objetivo: Minimizar costos totales
        def objective_rule(model):
            cost = 0
            for k in model.cars_set:
                vehicle = self.vehicles[k-1]
                for i in model.nodes_set:
                    for j in model.nodes_set:
                        cost += vehicle.calculate_cost(i, j, model.x[i, j, k],
                                                        self.car_matrix_distance,
                                                        self.car_matrix_time,
                                                        self.drone_matrix_distance,
                                                        num_depots)
            # Agregar costos de mantenimiento
            for k in model.cars_set:
                vehicle = self.vehicles[k-1]
                cost += vehicle.daily_maintenance
            return cost
        model.obj = Objective(rule=objective_rule, sense=minimize)

        # Restricciones

        # 1. Conservación de flujo para clientes
        def flow_conservation_rule(model, j, k):
            return sum(model.x[i, j, k] for i in model.nodes_set) - sum(model.x[j, i, k] for i in model.nodes_set) == 0
        model.flow_conservation = Constraint(model.clients_set, model.cars_set, rule=flow_conservation_rule)

        # 2. Cada cliente es visitado exactamente una vez
        def visit_once_rule(model, j):
            return sum(model.x[i, j, k] for i in model.nodes_set for k in model.cars_set) == 1
        model.visit_once = Constraint(model.clients_set, rule=visit_once_rule)

        # 3. Cada vehículo comienza y termina en un depósito
        def start_depot_rule(model, k):
            return sum(model.x[i, j, k] for i in model.depots_set for j in model.nodes_set) == 1
        model.start_depot = Constraint(model.cars_set, rule=start_depot_rule)

        def end_depot_rule(model, k):
            return sum(model.x[i, j, k] for j in model.nodes_set for i in model.depots_set) == 1
        model.end_depot = Constraint(model.cars_set, rule=end_depot_rule)

        # 4. Restricción de capacidad de los vehículos
        def capacity_rule(model, k):
            vehicle = self.vehicles[k-1]
            return sum(self.clients[j - num_depots - 1].product * model.x[i, j, k]
                       for i in model.nodes_set
                       for j in model.clients_set) <= vehicle.capacity
        model.capacity_constraint = Constraint(model.cars_set, rule=capacity_rule)

        # 5. Subtours eliminados (MTZ constraints)
        def subtour_elimination_rule(model, i, j, k):
            if i != j:
                return model.u[i, k] - model.u[j, k] + len(model.clients_set) * model.x[i, j, k] <= len(model.clients_set) - 1
            else:
                return Constraint.Skip
        model.subtour_elimination = Constraint(model.clients_set, model.clients_set, model.cars_set, rule=subtour_elimination_rule)

        # 6. Prohibir viajes de depósito a depósito
        def no_depot_to_depot_rule(model, i, j, k):
            if i <= len(self.depots) and j <= len(self.depots):
                return model.x[i, j, k] == 0
            else:
                return Constraint.Skip
        model.no_depot_to_depot = Constraint(model.nodes_set, model.nodes_set, model.cars_set, rule=no_depot_to_depot_rule)

        # 7. Restricción de rango
        def range_constraint_rule(model, k):
            vehicle = self.vehicles[k-1]
            total_distance = sum(
                vehicle.calculate_distance(i, j, self.car_matrix_distance, self.drone_matrix_distance) * model.x[i, j, k]
                for i in model.nodes_set for j in model.nodes_set
            )
            return total_distance <= vehicle.range_km
        model.range_constraint = Constraint(model.cars_set, rule=range_constraint_rule)

        """
        # 8. Restricción de capacidad en depósitos
        def depot_capacity_rule(model, i):
            depot = self.depots[i - 1]  # Obtener depósito correspondiente
            return sum(self.clients[j - num_depots].product * model.x[i, j, k]
                       for j in model.clients_set
                       for k in model.cars_set) <= depot.capacity
        model.depot_capacity_constraint = Constraint(model.depots_set, rule=depot_capacity_rule)
        """

        """
        def vehicle_and_depot_capacity_rule(model, k):
            # Obtener el vehículo
            vehicle = self.vehicles[k - 1]

            # Suma de demandas de los clientes atendidos por este vehículo
            total_demand = sum(self.clients[j - 1].product * model.x[i, j, k]
                               for i in model.depots_set  # Desde un depósito
                               for j in model.clients_set  # Hacia un cliente
            )

            # Capacidad del depósito desde donde sale el vehículo
            total_depot_capacity = sum(
                self.depots[i - 1].capacity * model.x[i, j, k]
                for i in model.depots_set
                for j in model.clients_set
            )

            # Restricción: suma de demandas <= capacidad del vehículo Y del depósito
            return total_demand <= min(vehicle.capacity, total_depot_capacity)
        model.vehicle_and_depot_capacity_constraint = Constraint(model.cars_set, rule=vehicle_and_depot_capacity_rule)
        """
        """
        def depot_capacity_rule(model, i):
            depot = self.depots[i - 1]  # Asegúrate de que i-1 es un índice válido
            return sum(
                self.clients[j - len(self.depots)].product * model.x[i, j, k]
                for j in model.clients_set
                for k in model.cars_set
                if j - len(self.depots) >= 0 and j - len(self.depots) < len(self.clients)
            ) <= depot.capacity
        model.depot_capacity_rule = Constraint(model.depots_set, rule=depot_capacity_rule)
        """

        def depot_cap_rule(model, i):
          for k in model.cars_set:
            sum(model.x[i,j,k] * self.Clients[j].product for j in model.clients_set) <= i.capacity
        model.depot_cap_rule = Constraint(model.depots_set, rule=depot_cap_rule)

        """
            depot = self.depots[i - 1]  # Asegúrate de que i-1 es un índice válido
            return sum(
                self.clients[j - len(self.depots)].product * model.x[i, j, k]
                for j in model.clients_set
                for k in model.cars_set
                if j - len(self.depots) >= 0 and j - len(self.depots) < len(self.clients)
            ) <= depot.capacity
        model.depot_capacity_rule = Constraint(model.depots_set, rule=depot_capacity_rule)
        """

        """
        idea adrian
        def dep_cap_rule(model, i):
          return sum(model.x[i,j,k]*  for j in clients_set for i in nodos_set for k in )
        """
    def solve_model(self):
      #solver = SolverFactory('glpk')
      solver_name = "highs"
      solver = SolverFactory(solver_name+"nl", executable=modules.find(solver_name), solve_io="nl")
      solver.options['time_limit'] = 1800
      result = solver.solve(self.model, tee=True)

      if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
          print("\nSolución óptima encontrada.")

          # Imprimir el costo total
          total_cost = value(self.model.obj)
          print(f"Costo total: {total_cost:,.2f}\n")

          # Imprimir las rutas y costos de cada vehículo
          for k in self.model.cars_set:
              vehicle = self.vehicles[k-1]
              print(f"Vehículo {k} ({vehicle.name}):")

              # Reconstruir la ruta para el vehículo k
              route = []
              num_depots = len(self.depots)
              print(num_depots)
              vehicle_cost = vehicle.daily_maintenance  # Incluir costo de mantenimiento diario

              # Encontrar el depósito de inicio para el vehículo k
              start_node = None
              for i in self.model.depots_set:
                  for j in self.model.nodes_set:
                      if value(self.model.x[i, j, k]) > 0.5:
                          start_node = j
                          route.append(str(self.index_to_location_id[i]))  # Agregar el depósito inicial
                          break
                  if start_node is not None:
                      break

              if start_node is None:
                  print("  No se encontró una ruta para este vehículo.")
                  continue

              current_node = start_node
              visited_nodes = set()
              visited_nodes.add(current_node)
              route.append(str(self.index_to_location_id[current_node]))

              while True:
                  next_node = None
                  for j in self.model.nodes_set:
                      if value(self.model.x[current_node, j, k]) > 0.5:
                          next_node = j
                          break
                  if next_node is None:
                      break
                  if next_node in visited_nodes:
                      # Verificar si hemos regresado a un depósito
                      if next_node in self.model.depots_set:
                          route.append(str(self.index_to_location_id[next_node]))

                          # Calcular costo del último arco
                          arc_cost = vehicle.calculate_cost(current_node, next_node, 1,self.car_matrix_distance,self.car_matrix_time,self.drone_matrix_distance, num_depots)
                          vehicle_cost += arc_cost
                      break
                  # Agregar nodo a la ruta
                  route.append(str(self.index_to_location_id[next_node]))
                  visited_nodes.add(next_node)

                  # Calcular costo del arco actual
                  arc_cost = vehicle.calculate_cost(current_node, next_node, 1,self.car_matrix_distance,self.car_matrix_time,self.drone_matrix_distance,num_depots)

                  vehicle_cost += arc_cost
                  current_node = next_node

              # Imprimir la ruta
              formatted_route = " -> ".join(route)
              print(f"  Ruta: {formatted_route}")
              print(f"  Costo: {vehicle_cost:,.2f}\n")

      elif result.solver.termination_condition == TerminationCondition.infeasible:
          print("No se encontró una solución factible.")
      else:
          print("El solver terminó con condición:", result.solver.termination_condition)

          # Imprimir el costo total
          total_cost = value(self.model.obj)
          print(f"Costo total: {total_cost:,.2f}\n")

          # Imprimir las rutas y costos de cada vehículo
          for k in self.model.cars_set:
              vehicle = self.vehicles[k-1]
              print(f"Vehículo {k} ({vehicle.name}):")

              # Reconstruir la ruta para el vehículo k
              route = []
              num_depots = len(self.depots)
              vehicle_cost = vehicle.daily_maintenance  # Incluir costo de mantenimiento diario

              # Encontrar el depósito de inicio para el vehículo k
              start_node = None
              for i in self.model.depots_set:
                  for j in self.model.nodes_set:
                      if value(self.model.x[i, j, k]) > 0.5:
                          start_node = j
                          route.append(str(self.index_to_location_id[i]))  # Agregar el depósito inicial
                          break
                  if start_node is not None:
                      break

              if start_node is None:
                  print("  No se encontró una ruta para este vehículo.")
                  continue

              current_node = start_node
              visited_nodes = set()
              visited_nodes.add(current_node)
              route.append(str(self.index_to_location_id[current_node]))

              while True:
                  next_node = None
                  for j in self.model.nodes_set:
                      if value(self.model.x[current_node, j, k]) > 0.5:
                          next_node = j
                          break
                  if next_node is None:
                      break
                  if next_node in visited_nodes:
                      # Verificar si hemos regresado a un depósito
                      if next_node in self.model.depots_set:
                          route.append(str(self.index_to_location_id[next_node]))

                          # Calcular costo del último arco
                          arc_cost = vehicle.calculate_cost(current_node, next_node, 1,self.car_matrix_distance,self.car_matrix_time,self.drone_matrix_distance, num_depots)
                          vehicle_cost += arc_cost
                      break
                  # Agregar nodo a la ruta
                  route.append(str(self.index_to_location_id[next_node]))
                  visited_nodes.add(next_node)

                  # Calcular costo del arco actual
                  arc_cost = vehicle.calculate_cost(current_node, next_node, 1,self.car_matrix_distance,self.car_matrix_time,self.drone_matrix_distance,num_depots)

                  vehicle_cost += arc_cost
                  current_node = next_node

              # Imprimir la ruta
              formatted_route = " -> ".join(route)
              print(f"  Ruta: {formatted_route}")
              print(f"  Costo: {vehicle_cost:,.2f}\n")
