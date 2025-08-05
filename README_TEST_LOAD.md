Utilizzo Base
bash
python test_load.py [cicli] [--gpu]
Parametri:

cicli: Numero di cicli di carico (default: 5)

--gpu: Abilita il carico sulla GPU 

Esempi Pratici
1. Test base con 3 cicli (solo CPU)
bash
python test_load.py 3

2. Test completo con 5 cicli (CPU + GPU)
bash
python test_load.py 5 --gpu

3. Test lungo con 10 cicli (CPU + GPU)
bash
python test_load.py 10 --gpu