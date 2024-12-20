import pandas as pd
import math as mt
from pyomo.environ import *

# ----------------------------------------------------------------------
# Cargar csv's
# ----------------------------------------------------------------------
path = 'case_1_base/'
clients = pd.read_csv(path + 'multi_clients.csv')
depots = pd.read_csv(path + 'Depots.csv')
vehicle = pd.read_csv(path + 'multi_vehicles.csv')

# ----------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------
nodos = [f"nodo{n+1}" for n in range(len(clients) + len(depots))]
vehiculos = [f"V{n+1}" for n in range(len(vehicle))]

# ----------------------------------------------------------------------
# Funciones de calculos
# ----------------------------------------------------------------------

def distancia_euclidiana(coord1, coord2):
    """Calcular la distancia euclidiana entre dos coordenadas."""
    return mt.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)

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
        cordenadas[nodo] = (float(depots.iloc[i, 1]), float(depots.iloc[i, 2]))

    return cordenadas

def get_matriz_distancias(nodos, cordenadas, len_clients):
    """Generar un diccionario anidado de distancias entre nodos."""
    distancias = {nodo_i: {} for nodo_i in nodos}
    
    for i, nodo_i in enumerate(nodos):
        for j, nodo_j in enumerate(nodos):
            if i != j:
                if not (i >= len_clients and j >= len_clients):
                    coord1 = cordenadas[nodo_i]
                    coord2 = cordenadas[nodo_j]
                    distancias[nodo_i][nodo_j] = round(distancia_euclidiana(coord1, coord2), 4)
                else:
                    distancias[nodo_i][nodo_j] = None # Aca es cuando es de un deposito a un deposito
            else:
                distancias[nodo_i][nodo_j] = 0
    
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
distancias = get_matriz_distancias(nodos, cordenadas, len(clients))
tipo = get_tipo(vehiculos, vehicle)
capacidad = get_capacidad(vehiculos, vehicle)

costo_extra = 0

# ----------------------------------------------------------------------
# MODELO
# ----------------------------------------------------------------------

# Crear el modelo
model = ConcreteModel()

# Variables de decisión
model.x = Var(nodos, nodos, vehiculos, within=Binary)
model.y = Var(nodos, vehiculos, within=NonNegativeIntegers) # después le ponemos máximo; nos da la candidad recargada

# Función objetivo
Model.obj = Objective(expr = sum(Model.x[i,j,k]*costosMatrix[i,j] for i in nodos for j in nodos for k in vehiculos) + sum(Model.y[i,k]*costo[k]), sense=minimize) # costo de recarga debe ser por tipo o vehículo

### Restricciones
# Restricción de "or": una localización debe tener un sensor o una localización adyacente debe tenerlo
def restriccion_or(model, loc):
    if sens_locs[loc][sensor] == 1:
        return model.x[loc] + sum(model.x[adyacente] for adyacente in adyas[loc]) >= 1
    else:
        return Constraint.Skip
model.restriccion_or = Constraint(localizaciones, rule=restriccion_or)

# Esta restricción asegura que la demanda asignada a un vehículo no supere su capacidad.
def restriccion_capacidad_vehiculo(model, k):
    return sum(model.x[i, j, k] * demanda[j] for i in nodos for j in nodos) <= capacidad[vehiculos[k]][1]
model.restriccion_capacidad_vehiculo = Constraint(vehiculos, rule=restriccion_capacidad_vehiculo)

# Esta restricción garantiza que la distancia recorrida por un vehículo no supere su capacidad de distancia diaria.
def restriccion_distancia_maxima(model, k):
    return sum(model.x[i, j, k] * distancias[i][j] for i in nodos for j in nodos) <= capacidad[vehiculos[k]][0]  # Capacidad máxima de distancia
model.restriccion_distancia_maxima = Constraint(vehiculos, rule=restriccion_distancia_maxima)

# Esta restricción asegura que cada vehículo salga de un depósito para iniciar su ruta.
def restriccion_salida_nodo_sucursal(model, k):
    return sum(model.x[deposito, j, k] for deposito in idDeposito for j in nodos if deposito != j) == 1
model.restriccion_salida_nodo_sucursal = Constraint(vehiculos, rule=restriccion_salida_nodo_sucursal)

# Para evitar duplicaciones, no puede haber más de un vehículo viajando entre los mismos nodos.
def restriccion_ruta_unica(model, i, j):
    return sum(model.x[i, j, k] for k in vehiculos) <= 1
model.restriccion_ruta_unica = Constraint(nodos, nodos, rule=restriccion_ruta_unica)

# Tal vez es necesario generar unas variables auxiliares para evitar subrutas
#model.u = Var(nodos, vehiculos, within=NonNegativeReals) # estas serian en teoria 
def restriccion_eliminacion_subrutas(model, i, j, k):
    if i != j and i != 0 and j != 0:  # No se aplica al depósito
        return model.u[i, k] - model.u[j, k] + len(nodos) * model.x[i, j, k] <= len(nodos) - 1
    else:
        return Constraint.Skip
model.restriccion_eliminacion_subrutas = Constraint(nodos, nodos, vehiculos, rule=restriccion_eliminacion_subrutas)



# Resolver el modelo
solver = SolverFactory('glpk')
solver.solve(model)

# Mostrar la solución para el sensor actual
solucion_sensor = [loc for loc in localizaciones if model.x[loc].value == 1]
print(f"Solución para el sensor {sensor}: {solucion_sensor}")

















