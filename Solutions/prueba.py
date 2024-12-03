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
clients = pd.read_csv(path + 'Clients.csv')
depots = pd.read_csv(path + 'Depots.csv')
vehicle = pd.read_csv(path + 'multi_vehicles.csv')

# ----------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------
nodos = [f"nodo{n+1}" for n in range(len(clients) + len(depots))]
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
            return distancia
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

def get_matriz_distancias(nodos, cordenadas, len_clients):
    """Generar un diccionario de diccionarios anidado de distancias entre nodos."""
    distancias = {}

    for i, nodo_i in enumerate(nodos):
        for j, nodo_j in enumerate(nodos):
            for tipoo in range(2):
                if i != j:
                    if not (i >= len_clients and j >= len_clients):
                        coord1 = cordenadas[nodo_i]
                        coord2 = cordenadas[nodo_j]
                        if tipoo == 1:
                            #print( nodo_i, nodo_j, tipoo)
                            distancias[(nodo_i, nodo_j, tipoo)] = round(distancia_haversiana(coord1, coord2), 4) # Si es un dron, haversiana
                        else:
                            #print( nodo_i, nodo_j, tipoo)
                            dist = distancia_osrm(coord1, coord2)
                            #print(f"Distancia OSRM: {dist}")
                            distancias[(nodo_i, nodo_j, tipoo)] = round(dist, 4) # Si es un carro, osrm
                    else:
                        distancias[(nodo_i, nodo_j, tipoo)] = None # Acá es cuando es de un depósito a un depósito
                else:
                    distancias[(nodo_i, nodo_j, tipoo)] = 0

    return distancias

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
distancias = get_matriz_distancias(nodos, cordenadas, len(clients))

# ----------------------------------------------------------------------
# Función y cálculo de distancias
# ----------------------------------------------------------------------






print(vehiculos)
print(tipo)
print(nodos)
print(demanda)
print("\n")
print(cordenadas)
print("\n")


print(distancias)








