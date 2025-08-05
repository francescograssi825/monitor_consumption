import pandas as pd

df = pd.read_csv('energy_monitor_20250804_123937.csv')

# Calcola consumi energetici totali
total_cpu_energy = df['power_consumption_watts'].sum() / 3600  # Wh
total_gpu_energy = df['gpu_0_power_watts'].sum() / 3600       # Wh

print(f"Consumo totale CPU: {total_cpu_energy:.2f} Wh")
print(f"Consumo totale GPU: {total_gpu_energy:.2f} Wh")