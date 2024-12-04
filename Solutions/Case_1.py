import numpy as np
import pandas as pd
import math as mt
from pyomo.environ import *

# ----------------------------------------------------------------------
# Imports para distancias carros y visualización en el mapa
# ----------------------------------------------------------------------
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import plotly.express as px

# ----------------------------------------------------------------------
# Cargar csv's
# ----------------------------------------------------------------------
path = 'Data/case_1_base/'
clients = pd.read_csv(path + 'ClientsMini.csv')
depots = pd.read_csv(path + 'DepotsMini.csv')
vehicle = pd.read_csv(path + 'VehiclesMini.csv')

# ----------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------
nodos = [f"nodo{n+1}" for n in range(len(depots) + len(clients))]
vehiculos = [f"V{n+1}" for n in range(len(vehicle))]

# ----------------------------------------------------------------------
# Funciones de cálculos
# ----------------------------------------------------------------------

def distancia_haversiana(coord1, coord2):
    """Calcular la distancia haversiana entre dos coordenadas."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371000  # radio de la Tierra en metros
    phi_1 = mt.radians(lat1)
    phi_2 = mt.radians(lat2)
    delta_phi = mt.radians(lat2 - lat1)
    delta_lambda = mt.radians(lon2 - lon1)
    a = mt.sin(delta_phi / 2.0) ** 2 + mt.cos(phi_1) * mt.cos(phi_2) * mt.sin(delta_lambda / 2.0) ** 2
    c = 2 * mt.atan2(mt.sqrt(a), mt.sqrt(1 - a))
    distancia = R * c
    return distancia

def distancia_osrm(coord1, coord2):
    """Calcular la distancia entre dos coordenadas usando OSRM."""
    #print(f"Coord1 (lon, lat): {coord1}")
    #print(f"Coord2 (lon, lat): {coord2}")
    
    start = "{},{}".format(coord1[0], coord1[1])
    end = "{},{}".format(coord2[0], coord2[1])
    
    url = 'http://router.project-osrm.org/route/v1/driving/{};{}?alternatives=false&annotations=nodes'.format(start, end)
    response = requests.get(url)
    
    # Print the response content for debugging
    #print("Response Content:", response.text)
    
    try:
        data = response.json()
        if data['code'] == 'Ok':
            distancia = data['routes'][0]['legs'][0]['distance']
            duracion = data['routes'][0]['legs'][0]['duration']
            return distancia, duracion
        else:
            print("Error in response:", data)
            return None
    except ValueError as e:
        print("JSON Parse Error:", e)
        return None

# ----------------------------------------------------------------------
# Funciones para parámetros nodos
# ----------------------------------------------------------------------
def get_demanda(nodos, clients):
    """Generar el diccionario de demanda para cada nodo."""
    return {
        nodo: (int(clients.iloc[i, 2]) if i < len(clients) else 0)
        for i, nodo in enumerate(nodos)
    }

def get_id_cliente(nodos, clients):
    """Generar el diccionario de IDs de clientes."""
    return {
        nodo: int(clients.iloc[i, 0])
        for i, nodo in enumerate(nodos) if i < len(clients)
    }

def get_id_deposito(nodos, depots, len_clients):
    """Generar el diccionario de IDs de depósitos."""
    return {
        nodo: int(depots.iloc[i - len_clients, 0])
        for i, nodo in enumerate(nodos) if i >= len_clients
    }

def get_coordenadas(nodos, clients, depots, len_clients):
    """Generar el diccionario de coordenadas para cada nodo."""
    cordenadas = {}
    # Agregar coordenadas de clientes
    for i, nodo in enumerate(nodos):
        if i < len_clients:
            cordenadas[nodo] = (float(clients.iloc[i, 3]), float(clients.iloc[i, 4]))
    
    # Agregar coordenadas de depósitos con valores predeterminados
    for i, nodo in enumerate(nodos[len_clients:], start=0):
        cordenadas[nodo] = (float(depots.iloc[i, 2]), float(depots.iloc[i, 3]))

    return cordenadas

def get_matriz_dist_costos(nodos, cordenadas, len_clients):
    """Generar un diccionario de diccionarios anidado de distancias entre nodos."""
    costos = {}
    distancias = {}

    for i, nodo_i in enumerate(nodos):
        for j, nodo_j in enumerate(nodos):
            for tipoo in range(2):
                if i != j:
                    if not (i >= len_clients and j >= len_clients):
                        coord1 = cordenadas[nodo_i]
                        coord2 = cordenadas[nodo_j]
                        if tipoo == 1:
                            dist = round(distancia_haversiana(coord1, coord2), 4)
                            costo1 = dist * (1/2) #  (dist / 1000) * 500 = dist * (500/1000)
                            tiempo = dist / 40 # distancia / velocidad = tiempo
                            costo2 = 500 * tiempo
                            costos[(nodo_i, nodo_j, tipoo)] = costo1 + costo2
                            distancias[(nodo_i, nodo_j, tipoo)] = dist
                        else:
                            dist, duracion = distancia_osrm(coord1, coord2)
                            costo1 = round(dist, 4) * 5 #  (dist / 1000) * 5000 = dist * (5000/1000)
                            costo2 = duracion * 500 * (1/60)
                            c_total = costo1 + costo2
                            costos[(nodo_i, nodo_j, tipoo)] = c_total
                            distancias[(nodo_i, nodo_j, tipoo)] = round(dist,4)
                    else:
                        costos[(nodo_i, nodo_j, tipoo)] = 9999999999999 # Acá es cuando es de un depósito a un depósito
                        distancias[(nodo_i,nodo_j, tipoo)] = 99999999999999
                else:
                    costos[(nodo_i, nodo_j, tipoo)] = 0
                    distancias[(nodo_i, nodo_j, tipoo)] = 0

    return distancias, costos

# ----------------------------------------------------------------------
# Funciones para parámetros Vehículos
# ----------------------------------------------------------------------
def get_tipo(vehiculos, vehicle):
    """Generar el diccionario de tipo para cada vehículo."""
    return {
        vehiculo: 1 if vehicle.iloc[i, 0] == 'drone' else 0
        for i, vehiculo in enumerate(vehiculos)
    }

def get_capacidad(vehiculos, vehicle):
    """Generar el diccionario de capacidad para cada vehículo."""
    return {
        vehiculo: (int(vehicle.iloc[i, 1]), int(vehicle.iloc[i, 2]))
        for i, vehiculo in enumerate(vehiculos)
    }

# ----------------------------------------------------------------------
# Calcular parámetros
# ----------------------------------------------------------------------
demanda = get_demanda(nodos, clients)
idCliente = get_id_cliente(nodos, clients)
idDeposito = get_id_deposito(nodos, depots, len(clients))
cordenadas = get_coordenadas(nodos, clients, depots, len(clients))
tipo = get_tipo(vehiculos, vehicle)
capacidad = get_capacidad(vehiculos, vehicle)
distancias, costos = get_matriz_dist_costos(nodos, cordenadas, len(clients))

# ----------------------------------------------------------------------
# MODELO
# ----------------------------------------------------------------------

# Crear el modelo
model = ConcreteModel()

# Variables de decisión
model.x = Var(nodos, nodos, vehiculos, within=Binary)

# Definir la variable u para eliminación de subrutas
model.u = Var(nodos, vehiculos, domain=NonNegativeIntegers, bounds=(1, len(nodos) - 1))


# Función objetivo
def objetivo(model):
    return sum(model.x[i, j, v] * costos[(i, j, tipo[v])] for i in nodos for j in nodos for v in vehiculos)
model.obj = Objective(rule=objetivo, sense=minimize)

### Restricciones
def restriccion_capacidad_vehiculo(model, k):
    # Ajuste para asegurar que estamos utilizando las claves correctamente
    return sum(model.x[i, j, k] * demanda[j] for i in nodos for j in nodos) <= capacidad[vehiculos[k]][0]


def restriccion_distancia_maxima(model, k):
    # Itera sobre los nodos por sus nombres
    return sum(model.x[i, j, k] * costos[(i, j, 0)] for i in nodos for j in nodos) <= capacidad[vehiculos[k]][0]


# Restricción de Salida del Depósito
def restriccion_salida_nodo_deposito(model, k):
    return sum(model.x[deposito, j, k] for deposito in idDeposito for j in nodos if deposito != j) == 1
model.restriccion_salida_nodo_deposito = Constraint(vehiculos, rule=restriccion_salida_nodo_deposito)


# Restricción de Ruta Única
def restriccion_ruta_unica(model, i, j):
    return sum(model.x[i, j, k] for k in vehiculos) <= 1
model.restriccion_ruta_unica = Constraint(nodos, nodos, rule=restriccion_ruta_unica)


# Restricción de eliminación de subrutas
def restriccion_eliminacion_subrutas(model, i, j, k):
    if i != j and i != 0 and j != 0:  # No se aplica al depósito
        return model.u[i, k] - model.u[j, k] + len(nodos) * model.x[i, j, k] <= len(nodos) - 1
    else:
        return Constraint.Skip
model.restriccion_eliminacion_subrutas = Constraint(nodos, nodos, vehiculos, rule=restriccion_eliminacion_subrutas)

"""
def restriccion_flujo_demanda(model, j):
    return sum(model.x[i, j, k] * demanda[j] for i in nodos for k in vehiculos) == demanda[j]
model.restriccion_flujo_demanda = Constraint(nodos, rule=restriccion_flujo_demanda)
"""

# Resolver el modelo
solver = SolverFactory('glpk')
# Resolver el modelo
solver_status = solver.solve(model, tee=True)

# Verificar si el modelo se resolvió correctamente
if solver_status.solver.termination_condition == TerminationCondition.optimal:
    print("La solución óptima fue encontrada.")
else:
    print("No se encontró una solución óptima.")

# Mostrar las rutas seleccionadas
print("\nRutas seleccionadas por el modelo:")
for k in vehiculos:
    print(f"\nVehículo {k}:")
    for i in nodos:
        for j in nodos:
            if model.x[i, j, k].value > 0.5:  # Considera que es 1 si el valor es mayor que 0.5 (siendo binaria)
                print(f"  - De {i} a {j}")

# Mostrar las posiciones de los nodos en el recorrido
"""
print("\nPosiciones de los nodos en el recorrido:")
for k in vehiculos:
    print(f"\nVehículo {k}:")
    for i in nodos:
        print(f"  Nodo {i}: Posición {model.u[i, k].value}")
        """

# Mostrar el valor de la función objetivo (costo total)
model.display()

















