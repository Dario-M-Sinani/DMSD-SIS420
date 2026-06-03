import gymnasium as gym

def main():
    # Inicializar el entorno de CliffWalking
    # render_mode="human" permite visualizar el agente moviéndose en el entorno
    env = gym.make('CliffWalking-v1', render_mode="human")

    # Reiniciar el entorno al estado inicial
    state, info = env.reset()
    done = False
    
    print("Iniciando Cliff Walking...")
    print(f"Estado inicial: {state}")

    # Ejecutar acciones aleatorias hasta que termine el episodio
    while not done:
        # Elegir una acción aleatoria: 0: Arriba, 1: Derecha, 2: Abajo, 3: Izquierda
        action = env.action_space.sample()
        
        # Tomar la acción en el entorno
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # El episodio termina si el agente llega a la meta o cae por el acantilado
        done = terminated or truncated
        
        print(f"Acción tomada: {action}, Nuevo estado: {next_state}, Recompensa: {reward}")

    print("Episodio finalizado.")
    env.close()

if __name__ == "__main__":
    main()
