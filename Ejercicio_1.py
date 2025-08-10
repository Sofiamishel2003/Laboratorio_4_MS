import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
import heapq

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Tipos para legibilidad (mantengo etiquetas cortas S/E/I/R para estados)
Estado = Literal["S","E","I","R"]
Rol    = Literal["paciente","personal"]

@dataclass
class Agente:
    """Representa a un individuo (paciente o personal) en una sala del hospital."""
    id: int
    rol: Rol
    sala: int
    estado: Estado = "S"
    aislado: bool = False
    dias_en_estado: int = 0  # usado por el modelo discreto

@dataclass
class Parametros:
    """Parámetros globales del hospital y de la enfermedad."""
    n_salas: int = 3
    pacientes_por_sala: int = 15
    personal_por_sala: int = 8
    dias_incubacion: int = 2
    dias_infeccioso: int = 5
    # Betas: probabilidad por "fuente infecciosa" efectiva (aprox. discreta)
    beta_pp: float = 0.05  # paciente->paciente
    beta_ps: float = 0.03  # paciente->personal
    beta_sp: float = 0.02  # personal->paciente
    beta_ss: float = 0.01  # personal->personal
    # Aislamiento y pruebas
    efect_aislamiento: float = 0.6   # reducción de transmisión si fuente está aislada
    sens_prueba: float = 0.85
    tasa_prueba_diaria_pac: float = 0.05
    tasa_prueba_diaria_per: float = 0.05
    # Tasas del modelo continuo (procesos de Poisson por día)
    tasa_contacto_paciente: float = 3.0
    tasa_contacto_personal: float = 4.0
    tasa_prueba_poisson_pac: float = 0.05
    tasa_prueba_poisson_per: float = 0.05

def inicializar_agentes(p: Parametros) -> List[Agente]:
    """Crea la población de agentes distribuidos por sala y rol."""
    agentes: List[Agente] = []
    aid = 0
    for s in range(p.n_salas):
        for _ in range(p.pacientes_por_sala):
            agentes.append(Agente(aid, "paciente", s)); aid += 1
        for _ in range(p.personal_por_sala):
            agentes.append(Agente(aid, "personal", s)); aid += 1
    return agentes

def sembrar_infecciones(agentes: List[Agente], n_inicial_I=6):
    """Marca aleatoriamente algunos agentes como infecciosos iniciales."""
    idx = np.random.choice(len(agentes), n_inicial_I, replace=False)
    for i in idx:
        agentes[i].estado = "I"

# MODELO 1: Tiempo discreto (actualizaciones diarias)
def correr_discreto(dias: int = 40, p: Parametros = Parametros()) -> Tuple[pd.DataFrame, List[Agente]]:
    """Simula día a día con actualizaciones síncronas de estados."""
    agentes = inicializar_agentes(p)
    sembrar_infecciones(agentes, n_inicial_I=6)

    def beta_efectiva(fuente_aislada: bool, beta: float) -> float:
        """Reduce beta si la fuente está aislada."""
        return beta * (1 - p.efect_aislamiento if fuente_aislada else 1.0)

    metricas = []
    for d in range(dias):
        # Agrupar por sala y rol
        salas = {s: {"paciente": [], "personal": []} for s in range(p.n_salas)}
        for a in agentes:
            salas[a.sala][a.rol].append(a)

        nuevos_expuestos: List[Agente] = []

        # Transmisión por sala
        for s in range(p.n_salas):
            pac = salas[s]["paciente"]; per = salas[s]["personal"]
            I_pac = [a for a in pac if a.estado == "I"]
            I_per = [a for a in per if a.estado == "I"]

            # Pacientes susceptibles
            for a in pac:
                if a.estado != "S":
                    continue
                # Riesgo agregado por cada infeccioso (aprox. hazard discreto)
                lam = sum(beta_efectiva(src.aislado, p.beta_pp) for src in I_pac) + \
                      sum(beta_efectiva(src.aislado, p.beta_sp) for src in I_per)
                prob = 1 - np.exp(-lam)
                if np.random.rand() < prob:
                    nuevos_expuestos.append(a)

            # Personal susceptible
            for a in per:
                if a.estado != "S":
                    continue
                lam = sum(beta_efectiva(src.aislado, p.beta_ps) for src in I_pac) + \
                      sum(beta_efectiva(src.aislado, p.beta_ss) for src in I_per)
                prob = 1 - np.exp(-lam)
                if np.random.rand() < prob:
                    nuevos_expuestos.append(a)

        # Aplicar nuevas exposiciones (S->E)
        for a in nuevos_expuestos:
            a.estado = "E"; a.dias_en_estado = 0

        # Pruebas diarias (Bernoulli) y aislamiento de I detectados
        for a in agentes:
            tasa = p.tasa_prueba_diaria_pac if a.rol == "paciente" else p.tasa_prueba_diaria_per
            if a.estado == "I" and np.random.rand() < tasa and np.random.rand() < p.sens_prueba:
                a.aislado = True

        # Progresión de estados por umbral de días
        for a in agentes:
            a.dias_en_estado += 1
            if a.estado == "E" and a.dias_en_estado >= p.dias_incubacion:
                a.estado = "I"; a.dias_en_estado = 0
            elif a.estado == "I" and a.dias_en_estado >= p.dias_infeccioso:
                a.estado = "R"; a.dias_en_estado = 0; a.aislado = False

        # Registrar métricas del día
        metricas.append({
            "dia": d + 1,
            "S": sum(1 for x in agentes if x.estado == "S"),
            "E": sum(1 for x in agentes if x.estado == "E"),
            "I": sum(1 for x in agentes if x.estado == "I"),
            "R": sum(1 for x in agentes if x.estado == "R"),
            "aislados": sum(1 for x in agentes if x.aislado),
            "nuevos_E": len(nuevos_expuestos),
        })

    df = pd.DataFrame(metricas)
    return df, agentes

df_dis, agentes_dis = correr_discreto(dias=40)
print("Métricas (Tiempo discreto, actualizaciones diarias)")
print(df_dis.head(10).to_string(index=False))

plt.figure()
plt.plot(df_dis["dia"], df_dis["I"], label="Infecciosos")
plt.plot(df_dis["dia"], df_dis["aislados"], label="Aislados")
plt.title("Tiempo discreto: Infecciosos y Aislados")
plt.xlabel("Día")
plt.ylabel("Cantidad")
plt.legend()
plt.show()

ruta_csv_dis = "data/metricas_discreto_hospital.csv"
df_dis.to_csv(ruta_csv_dis, index=False)

# MODELO 2: Tiempo continuo (impulsado por eventos con cola de prioridad)
TipoEvento = Literal["contacto","fin_incubacion","recuperacion","prueba"]

@dataclass(order=True)
class Evento:
    """Evento con orden natural por tiempo (y prioridad en empates)."""
    tiempo: float
    prioridad: int
    tipo: TipoEvento
    quien: Optional[int] = field(compare=False, default=None)   # agente principal
    otro: Optional[int] = field(compare=False, default=None)    # agente secundario (si aplica)
    sala: Optional[int] = field(compare=False, default=None)    # sala involucrada (si aplica)

def correr_continuo(Tmax_dias: float = 40.0, p: Parametros = Parametros()) -> Tuple[pd.DataFrame, List[Agente]]:
    """Simula con reloj continuo y cola de eventos (contactos, cambios de estado, pruebas)."""
    agentes = inicializar_agentes(p)
    sembrar_infecciones(agentes, n_inicial_I=6)

    reloj = 0.0
    pq: List[Evento] = []  # cola de prioridad por tiempo y prioridad

    def agendar(tipo: TipoEvento, t: float, prioridad: int, quien=None, otro=None, sala=None):
        """Insertar un nuevo evento en la cola."""
        heapq.heappush(pq, Evento(t, prioridad, tipo, quien, otro, sala))

    def espera_exp(rate: float) -> float:
        """Tiempo de espera exponencial con tasa 'rate' por día."""
        return np.random.exponential(1.0 / rate) if rate > 0 else np.inf

    # Agendar pruebas de forma inicial (proceso de Poisson por agente)
    for a in agentes:
        tasa = p.tasa_prueba_poisson_pac if a.rol == "paciente" else p.tasa_prueba_poisson_per
        t = reloj + espera_exp(tasa)
        if t < np.inf:
            agendar("prueba", t, 2, quien=a.id)

    # Para infecciosos iniciales: agendar recuperación y primer contacto
    def agendar_si_infeccioso(a: Agente):
        if a.estado != "I":
            return
        agendar("recuperacion", reloj + p.dias_infeccioso, 1, quien=a.id)
        tasa_c = p.tasa_contacto_paciente if a.rol == "paciente" else p.tasa_contacto_personal
        t = reloj + espera_exp(tasa_c)
        agendar("contacto", t, 3, quien=a.id, sala=a.sala)

    for a in agentes:
        if a.estado == "I":
            agendar_si_infeccioso(a)

    # Métricas: muestreo diario
    metricas = []
    tiempos_muestreo = np.linspace(0, Tmax_dias, int(Tmax_dias) + 1)
    sig_idx = 0

    def registrar(t: float):
        metricas.append({
            "tiempo": t,
            "S": sum(1 for x in agentes if x.estado == "S"),
            "E": sum(1 for x in agentes if x.estado == "E"),
            "I": sum(1 for x in agentes if x.estado == "I"),
            "R": sum(1 for x in agentes if x.estado == "R"),
            "aislados": sum(1 for x in agentes if x.aislado),
        })

    registrar(0.0)

    while pq and reloj <= Tmax_dias:
        ev = heapq.heappop(pq)
        if ev.tiempo > Tmax_dias:
            break

        # Avanzar el reloj
        reloj = ev.tiempo

        # Registrar métricas en los puntos de muestreo alcanzados
        while sig_idx < len(tiempos_muestreo) and tiempos_muestreo[sig_idx] <= reloj:
            registrar(tiempos_muestreo[sig_idx])
            sig_idx += 1

        # Procesar evento según tipo
        if ev.tipo == "contacto":
            src = agentes[ev.quien]
            if src.estado != "I":
                # Fuente ya no es infecciosa; descartar evento
                continue

            # Elegir un contacto aleatorio en la misma sala
            candidatos = [x for x in agentes if x.sala == src.sala and x.id != src.id]
            if not candidatos:
                continue
            tgt = np.random.choice(candidatos)

            # Calcular probabilidad de transmisión por contacto
            if tgt.estado == "S":
                if src.rol == "paciente" and tgt.rol == "paciente":
                    beta = p.beta_pp
                elif src.rol == "paciente" and tgt.rol == "personal":
                    beta = p.beta_ps
                elif src.rol == "personal" and tgt.rol == "paciente":
                    beta = p.beta_sp
                else:
                    beta = p.beta_ss
                if src.aislado:
                    beta = beta * (1 - p.efect_aislamiento)
                if np.random.rand() < beta:
                    # Exposición: programar fin de incubación
                    tgt.estado = "E"
                    agendar("fin_incubacion", reloj + p.dias_incubacion, 1, quien=tgt.id)

            # Programar el siguiente contacto de la fuente
            tasa_c = p.tasa_contacto_paciente if src.rol == "paciente" else p.tasa_contacto_personal
            t_sig = reloj + espera_exp(tasa_c)
            agendar("contacto", t_sig, 3, quien=src.id, sala=src.sala)

        elif ev.tipo == "fin_incubacion":
            a = agentes[ev.quien]
            if a.estado == "E":
                a.estado = "I"
                # Programar recuperación y primer contacto
                agendar("recuperacion", reloj + p.dias_infeccioso, 1, quien=a.id)
                tasa_c = p.tasa_contacto_paciente if a.rol == "paciente" else p.tasa_contacto_personal
                t_sig = reloj + espera_exp(tasa_c)
                agendar("contacto", t_sig, 3, quien=a.id, sala=a.sala)

        elif ev.tipo == "recuperacion":
            a = agentes[ev.quien]
            if a.estado == "I":
                a.estado = "R"
                a.aislado = False  # levantar aislamiento al recuperarse

        elif ev.tipo == "prueba":
            a = agentes[ev.quien]
            # Reprogramar la siguiente prueba (proceso de Poisson)
            tasa = p.tasa_prueba_poisson_pac if a.rol == "paciente" else p.tasa_prueba_poisson_per
            t_sig = reloj + espera_exp(tasa)
            agendar("prueba", t_sig, 2, quien=a.id)
            # Ejecución de la prueba
            if a.estado == "I" and np.random.rand() < p.sens_prueba:
                a.aislado = True

    # Asegurar último muestreo
    if sig_idx < len(tiempos_muestreo):
        for i in range(sig_idx, len(tiempos_muestreo)):
            registrar(tiempos_muestreo[i])

    df = pd.DataFrame(metricas).drop_duplicates(subset=["tiempo"]).sort_values("tiempo")
    return df, agentes

df_con, agentes_con = correr_continuo(Tmax_dias=40.0)
print("Métricas (Tiempo continuo, impulsado por eventos)")
print(df_con.head(10).to_string(index=False))

plt.figure()
plt.plot(df_con["tiempo"], df_con["I"], label="Infecciosos")
plt.plot(df_con["tiempo"], df_con["aislados"], label="Aislados")
plt.title("Tiempo continuo: Infecciosos y Aislados")
plt.xlabel("Tiempo (días)")
plt.ylabel("Cantidad")
plt.legend()
plt.show()

ruta_csv_con = "data/metricas_continuo_hospital.csv"
df_con.to_csv(ruta_csv_con, index=False)

# Guardar ambos CSV y dejar rutas disponibles
(ruta_csv_dis, ruta_csv_con)
