import sys
import subprocess
import time
import csv
import psutil
import threading
from datetime import datetime
import os
import signal

# Configurazione delle dipendenze opzionali
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    from codecarbon.core.units import Energy
    from codecarbon.external.logger import logger
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

if CODECARBON_AVAILABLE:
    logger.setLevel("ERROR")

class GPUEnergyMonitor:
    def __init__(self, sampling_rate=2, output_file=None, country_code="ITA"):
        self.sampling_rate = sampling_rate
        self.sampling_interval = 1.0 / sampling_rate
        self.monitoring = False
        self.data = []
        self.country_code = country_code
        
        # File di output
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"gpu_energy_{timestamp}.csv"
        else:
            self.output_file = output_file
            
        # Variabili di stato
        self.target_process = None
        self.target_pid = None
        self.monitor_thread = None
        self.start_time = None
        
        # Inizializza NVIDIA
        self.gpu_info = self.initialize_nvidia()
        
        # Inizializza CodeCarbon
        self.tracker = None
        self.codecarbon_started = False
        self.final_emissions = None
        self.final_energy = None
        self.final_cpu_energy = None
        self.final_gpu_energy = None
        self.final_ram_energy = None
        
        # Headers CSV 
        self.csv_headers = [
            'timestamp',
            'elapsed_time',
            'cpu_percent',
            'memory_used_gb',
            'gpu_name',
            'gpu_utilization',
            'gpu_memory_used_mb',
            'gpu_memory_percent',
            'gpu_temperature',
            'gpu_power_watts',
            'gpu_power_limit',
            'codecarbon_energy_kwh',
            'codecarbon_emissions_kg_co2',
            'codecarbon_power_watts'
        ]

    def initialize_nvidia(self):
        gpu_info = {
            'available': False,
            'gpus': [],
            'driver': "N/A"
        }
        
        if not NVIDIA_AVAILABLE:
            return gpu_info
        
        try:
            pynvml.nvmlInit()
            gpu_info['available'] = True
            
            # Gestione compatibilità per la versione del driver
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode()
            gpu_info['driver'] = driver_version
            
            gpu_count = pynvml.nvmlDeviceGetCount()
            print(f"Driver NVIDIA: {gpu_info['driver']}")
            print(f"GPU NVIDIA rilevate: {gpu_count}")
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                # Gestione sia stringhe che byte array
                if isinstance(name, bytes):
                    name = name.decode()
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory_mb = mem_info.total // (1024 * 1024)
                
                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] // 1000
                except:
                    power_limit = "N/A"
                
                gpu_data = {
                    'index': i,
                    'handle': handle,
                    'name': name,
                    'total_memory_mb': total_memory_mb,
                    'power_limit': power_limit
                }
                
                gpu_info['gpus'].append(gpu_data)
                print(f"GPU {i}: {name} ({total_memory_mb} MB)")
                
        except Exception as e:
            print(f"Errore inizializzazione NVIDIA: {e}")
            gpu_info['available'] = False
            
        return gpu_info

    def get_gpu_stats(self, gpu_index=0):
        """Ottiene le statistiche essenziali per una GPU"""
        if not self.gpu_info['available'] or gpu_index >= len(self.gpu_info['gpus']):
            return {}
        
        try:
            gpu = self.gpu_info['gpus'][gpu_index]
            handle = gpu['handle']

            # Inizializza il dizionario stats con valori di default
            stats = {
                'name': gpu['name'],
                'utilization': None,
                'memory_used_mb': None,
                'memory_percent': None,
                'temperature': None,
                'power_watts': None,
                'power_limit': gpu.get('power_limit')
            }
            
            # Utilizzo GPU
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats['utilization'] = util.gpu
            except:
                pass
            
            # Memoria
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats['memory_used_mb'] = mem_info.used // (1024 * 1024)
                stats['memory_percent'] = (mem_info.used / mem_info.total) * 100
            except:
                pass
            
            # Temperatura
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                stats['temperature'] = temp
            except pynvml.NVMLError as e:
                
                if e.value != pynvml.NVML_ERROR_NOT_SUPPORTED:
                    print(f"Errore temperatura GPU: {e}")
            except:
                pass
            
            # Consumo energetico
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW -> W
                stats['power_watts'] = power
            except:
                pass
            
            return stats
            
        except Exception as e:
            print(f"Errore lettura GPU: {e}")
            return {}

    def get_process_stats(self):
        """Ottiene statistiche del processo target"""
        if not self.target_pid:
            return None, None
            
        try:
            process = psutil.Process(self.target_pid)
            return process.cpu_percent(), process.memory_info().rss / (1024 * 1024)
        except:
            return None, None

    def get_codecarbon_metrics(self):
        """Ottiene metriche da CodeCarbon"""
        if not self.tracker or not self.codecarbon_started:
            return {}
        
        try:
            metrics = {
                'energy_kwh': 0,
                'emissions_kg_co2': 0,
                'power_watts': 0
            }
            
            # Accesso ai dati interni del tracker
            if hasattr(self.tracker, '_total_energy'):
                total_energy = self.tracker._total_energy
                # Estrazione dei valori numerici
                if isinstance(total_energy, Energy):
                    metrics['energy_kwh'] = total_energy.kWh
                else:
                    metrics['energy_kwh'] = total_energy
            
            if hasattr(self.tracker, '_total_emissions'):
                metrics['emissions_kg_co2'] = self.tracker._total_emissions

            
            
            # Calcola potenza istantanea
            current_time = time.time()
            current_energy = metrics['energy_kwh']
            
            if hasattr(self, 'last_cc_time') and hasattr(self, 'last_cc_energy'):
                time_diff = current_time - self.last_cc_time
                energy_diff = current_energy - self.last_cc_energy
                if time_diff > 0 and energy_diff >= 0:
                    metrics['power_watts'] = (energy_diff * 1000 * 3600) / time_diff
            
            self.last_cc_time = current_time
            self.last_cc_energy = current_energy
            
            return metrics
            
        except Exception as e:
            # Ritorna metriche vuote in caso di errore
            return {
                'energy_kwh': 0,
                'emissions_kg_co2': 0,
                'power_watts': 0
            }

    def collect_data(self):
        """Raccoglie tutti i dati essenziali"""
        timestamp = datetime.now().isoformat()
        elapsed_time = time.time() - self.start_time
        
        # Dati di sistema
        cpu_percent = psutil.cpu_percent()
        memory_used_gb = psutil.virtual_memory().used / (1024**3)
        
        # Dati GPU
        gpu_stats = self.get_gpu_stats(0)
        
        # Dati processo
        process_cpu, process_mem = self.get_process_stats()
        
        # Dati CodeCarbon
        cc_metrics = self.get_codecarbon_metrics() if CODECARBON_AVAILABLE else {}
        
        
        row_data = [
            timestamp,
            elapsed_time,
            cpu_percent,
            memory_used_gb,
            gpu_stats.get('name', 'N/A'),
            gpu_stats.get('utilization'),
            gpu_stats.get('memory_used_mb'),
            gpu_stats.get('memory_percent'),
            gpu_stats.get('temperature'),
            gpu_stats.get('power_watts'),
            gpu_stats.get('power_limit')
        ]
        
    
        if CODECARBON_AVAILABLE:
            row_data.extend([
                cc_metrics.get('energy_kwh', 0),
                cc_metrics.get('emissions_kg_co2', 0),
                cc_metrics.get('power_watts', 0)
            ])
        else:
           
            row_data.extend([None] * 3)
        
        return row_data

    def monitor_loop(self):
        """Loop principale di monitoraggio"""
        print(f"Avvio monitoraggio con frequenza {self.sampling_rate} Hz")
        print(f"Salvataggio dati in: {self.output_file}")
        
        # Inizializzazione file CSV
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.csv_headers)
        
        self.start_time = time.time()
        
        while self.monitoring:
            try:
                data_row = self.collect_data()
                
                # Salvataggio
                with open(self.output_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(data_row)
                
                # Stampa statistiche
                self.print_stats(data_row)
                
                self.data.append(data_row)
                
                time.sleep(self.sampling_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Errore monitoraggio: {type(e).__name__}: {e}")
                time.sleep(self.sampling_interval)

    def print_stats(self, data_row):
        """Stampa una sintesi delle statistiche"""
        elapsed = data_row[1]
        cpu_percent = data_row[2]
        gpu_util = data_row[5]
        gpu_temp = data_row[8]
        gpu_power = data_row[9]
        gpu_mem = data_row[7]
        
        stats_line = f"T: {elapsed:.1f}s | CPU: {cpu_percent:.1f}%"
        
        if gpu_util is not None:
            stats_line += f" | GPU: {gpu_util:.1f}%"
        if gpu_mem is not None:
            stats_line += f" | VRAM: {gpu_mem:.1f}%"
        if gpu_temp is not None:
            stats_line += f" | Temp: {gpu_temp:.1f}°C"
        if gpu_power is not None:
            stats_line += f" | GPU Power: {gpu_power:.1f}W"
        
        # Aggiunta dati CodeCarbon se disponibili
        if CODECARBON_AVAILABLE and len(data_row) > 13:
            cc_energy = data_row[11] or 0
            cc_emissions = data_row[12] or 0
            cc_power = data_row[13] or 0
            
            stats_line += f" | CC Energy: {cc_energy*1000:.4f}Wh"
            stats_line += f" | CC CO2: {cc_emissions*1000:.4f}g"
            stats_line += f" | CC Power: {cc_power:.1f}W"
        
        print(f"\r{stats_line}", end="", flush=True)

    def start_monitoring(self, target_command):
        """Avvia il monitoraggio e il processo target"""
        # Avvia CodeCarbon se disponibile
        if CODECARBON_AVAILABLE:
            print("Avvio CodeCarbon tracker...")
            
            # Crea la directory per i log se non esiste
            output_dir = "./codecarbon_logs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Creata directory per i log: {output_dir}")
            
           
            try:
                self.tracker = EmissionsTracker(
                    measure_power_secs=self.sampling_interval,
                    output_dir=output_dir,
                    country_iso_code=self.country_code, 
                    log_level="ERROR",
                    save_to_file=True,
                    tracking_mode="process",
                    gpu_ids=[0] if self.gpu_info['available'] else []
                )
            except:
               
                print("Usando modalità offline per CodeCarbon")
                self.tracker = OfflineEmissionsTracker(
                    country_iso_code=self.country_code,
                    output_dir=output_dir,
                    log_level="ERROR",
                    measure_power_secs=self.sampling_interval
                )
                
            self.tracker.start()
            self.codecarbon_started = True
            # Inizializza variabili per calcolo potenza
            self.last_cc_time = time.time()
            self.last_cc_energy = 0
        
        # Avvia il processo target
        print(f"Avvio processo: {' '.join(target_command)}")
        self.target_process = subprocess.Popen(
            target_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.target_pid = self.target_process.pid
        print(f"Processo avviato con PID: {self.target_pid}")
        
        # Avvia monitoraggio
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Aspetta che il processo finisca
        try:
            stdout, stderr = self.target_process.communicate()
            
            # Mostra output
            if stdout:
                print(f"\n--- Output del processo ---\n{stdout}")
            if stderr:
                print(f"\n--- Errori del processo ---\n{stderr}")
                
        except KeyboardInterrupt:
            print(f"\nInterruzione rilevata, terminando processo...")
            self.target_process.terminate()
            self.target_process.wait()
        
        # Ferma monitoraggio
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        # Ferma CodeCarbon e ottenimento risultati finali
        if CODECARBON_AVAILABLE and self.codecarbon_started:
            try:
                self.final_emissions = self.tracker.stop()
                self.final_energy = getattr(self.tracker, '_total_energy', 0)
                
                # Estrazione dei valori energetici finali per componente
                if hasattr(self.tracker, '_tracker'):
                    tracker = self.tracker._tracker
                    self.final_cpu_energy = getattr(tracker, '_cpu_energy', Energy(0)).kWh
                    self.final_gpu_energy = getattr(tracker, '_gpu_energy', Energy(0)).kWh
                    self.final_ram_energy = getattr(tracker, '_ram_energy', Energy(0)).kWh
                
                self.codecarbon_started = False
            except Exception as e:
                print(f"Errore durante la chiusura di CodeCarbon: {e}")
        
        print(f"\nMonitoraggio completato. Dati salvati in: {self.output_file}")
        print(f"Campioni raccolti: {len(self.data)}")
        
        # Riepilogo finale
        self.print_summary()

    def print_summary(self):
        """Stampa un riepilogo delle statistiche GPU e CodeCarbon"""
        if not self.data:
            return
            
        print("\n--- Riepilogo GPU ---")
        
        # Estrazione delle solo le colonne rilevanti
        gpu_util = [row[5] for row in self.data if row[5] is not None]
        gpu_temp = [row[8] for row in self.data if row[8] is not None]
        gpu_power = [row[9] for row in self.data if row[9] is not None]
        gpu_mem = [row[7] for row in self.data if row[7] is not None]
        
        if gpu_util:
            avg = sum(gpu_util) / len(gpu_util)
            max_val = max(gpu_util)
            print(f"Utilizzo GPU: Media={avg:.1f}%, Max={max_val:.1f}%")
        
        if gpu_mem:
            avg = sum(gpu_mem) / len(gpu_mem)
            max_val = max(gpu_mem)
            print(f"Memoria GPU: Media={avg:.1f}%, Max={max_val:.1f}%")
        
        if gpu_temp:
            avg = sum(gpu_temp) / len(gpu_temp)
            max_val = max(gpu_temp)
            print(f"Temperatura: Media={avg:.1f}°C, Max={max_val:.1f}°C")
        
        if gpu_power:
            avg = sum(gpu_power) / len(gpu_power)
            max_val = max(gpu_power)
            print(f"Potenza GPU: Media={avg:.1f}W, Max={max_val:.1f}W")
            
        # Durata totale
        duration = self.data[-1][1]
        print(f"Durata totale: {duration:.1f} secondi")
        
        # Riepilogo CodeCarbon
        if CODECARBON_AVAILABLE and self.final_energy is not None and self.final_emissions is not None:
            print("\n--- Riepilogo CodeCarbon ---")
            
            # Estrazione valore numerico se è un oggetto Energy
            if isinstance(self.final_energy, Energy):
                self.final_energy = self.final_energy.kWh
                
            print(f"Energia totale consumata: {self.final_energy:.6f} kWh")
            print(f"Emissioni totali CO2: {self.final_emissions:.6f} kg")
            
            # Stampa i consumi per componente
            if self.final_cpu_energy is not None:
                print(f"Energia CPU totale: {self.final_cpu_energy*1000:.4f} Wh")
            if self.final_gpu_energy is not None:
                print(f"Energia GPU totale: {self.final_gpu_energy*1000:.4f} Wh")
            if self.final_ram_energy is not None:
                print(f"Energia RAM totale: {self.final_ram_energy*1000:.4f} Wh")
        elif CODECARBON_AVAILABLE:
            print("\nAttenzione: Dati CodeCarbon incompleti. Verifica la connessione internet e la configurazione della regione.")

def main():
    if len(sys.argv) < 2:
        print("Utilizzo: python3 gpu_monitor.py [-f FREQ] [-c COUNTRY] nome_programma.py [args...]")
        print("Esempio: python3 gpu_monitor.py -f 5 -c ITA script.py")
        sys.exit(1)
    
    # Parametri configurabili
    sampling_rate = 2  # Default 2Hz
    target_start = 1
    country_code = "ITA"  # Default country
    
    # Parse arguments
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-f':
            try:
                sampling_rate = float(sys.argv[i+1])
                target_start = i+2
                i += 2
                print(f"Frequenza impostata a: {sampling_rate} Hz")
                continue
            except:
                print("Errore: frequenza non valida")
                sys.exit(1)
        elif sys.argv[i] == '-c':
            try:
                country_code = sys.argv[i+1]
                target_start = i+2
                i += 2
                print(f"Codice paese impostato a: {country_code}")
                continue
            except:
                print("Errore: codice paese non valido")
                sys.exit(1)
        i += 1
    
    # Comando target
    target_command = ['python3'] + sys.argv[target_start:]
    
    # Verifica NVIDIA
    if not NVIDIA_AVAILABLE:
        print("\nATTENZIONE: pynvml non installato!")
        print("Per il monitoraggio GPU installa: pip install pynvml")
        print("Monitoraggio limitato a CPU e memoria...\n")
    
    # Verifica CodeCarbon
    if not CODECARBON_AVAILABLE:
        print("\nATTENZIONE: codecarbon non installato!")
        print("Per il monitoraggio delle emissioni installa: pip install codecarbon")
        print("Dati di emissione non saranno disponibili...\n")
    
    # Crea e avvia monitor
    monitor = GPUEnergyMonitor(sampling_rate=sampling_rate, country_code=country_code)
    
    # Gestore segnali
    def signal_handler(signum, frame):
        print("\nTerminazione richiesta...")
        monitor.monitoring = False
        if monitor.target_process:
            monitor.target_process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        monitor.start_monitoring(target_command)
    except Exception as e:
        print(f"Errore: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()