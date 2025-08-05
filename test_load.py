import time
import math
import argparse
import numpy as np
import threading

def cpu_load(duration, intensity):
    """Genera carico CPU controllato"""
    end_time = time.time() + duration
    while time.time() < end_time:
        # Converti l'intensità in un intero per range()
        iterations = max(1, int(intensity * 1000))
        for _ in range(iterations):
            math.sqrt(math.fsum([x**2 for x in range(100)]))
        time.sleep(0.01)

def gpu_load(duration, intensity):
    """Genera carico GPU usando operazioni vettoriali"""
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        end_time = time.time() + duration
        while time.time() < end_time:
            size = max(1, int(intensity * 1000))
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            torch.matmul(a, b)
            time.sleep(0.01)
            
    except ImportError:
        # Fallback su operazioni CPU se PyTorch non disponibile
        print("PyTorch non installato - usando operazioni CPU per simulare GPU load")
        end_time = time.time() + duration
        while time.time() < end_time:
            size = max(1, int(intensity * 1000))
            a = np.random.randn(size, size)
            b = np.random.randn(size, size)
            np.dot(a, b)
            time.sleep(0.1)

def main():
    parser = argparse.ArgumentParser(description='Generatore di carico CPU/GPU')
    parser.add_argument('cicli', type=int, nargs='?', default=5, help='Numero di cicli di carico')
    parser.add_argument('--gpu', action='store_true', help='Abilita carico GPU')
    args = parser.parse_args()

    print(f"Avvio test con {args.cicli} cicli {'(CPU+GPU)' if args.gpu else '(solo CPU)'}")
    
    # Fase iniziale: stress GPU al 100% per 5 secondi (solo se abilitata la GPU)
    if args.gpu:
        print("\n--- FASE INIZIALE: STRESS GPU AL 100% PER 5 SECONDI ---")
        gpu_load(5, 1.0)
        print("--- FASE INIZIALE COMPLETATA ---\n")
        time.sleep(1) 
    
    for i in range(args.cicli):
        # Carico crescente
        intensity = (i + 1) / args.cicli
        
        # Fase CPU
        print(f"\nCiclo {i+1}/{args.cicli} - Carico CPU ({intensity*100:.0f}%)")
        cpu_load(5, intensity)
        
        # Fase GPU (se richiesto)
        if args.gpu:
            print(f"Ciclo {i+1}/{args.cicli} - Carico GPU ({intensity*100:.0f}%)")
            gpu_load(5, intensity)
        
        # Fase idle
        print(f"Ciclo {i+1}/{args.cicli} - Idle")
        time.sleep(2)
    
    print("\nTest completato")

if __name__ == "__main__":
    main()