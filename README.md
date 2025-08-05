# Installazione dipendenze
pip install codecarbon psutil pynvml

# Uso base
python3 monitor.py script.py

# Con frequenza personalizzata (2 campioni al secondo)
python3 monitor.py -f 2 -c ITA script.py arg1 arg2

# Esempio con script esistente
python3 monitor.py mio_script.py --gpu --epochs 100