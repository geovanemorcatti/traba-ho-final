# 1. Para o Fuzzy de MAMDANI (Baseado em Regras e Centroide)
import skfuzzy as fuzzy
import skfuzzy.control as ctrl  # <-- Linha oficial para criar o sistema de Mamdani

# 2. Para o Fuzzy de TAKAGI-SUGENO / SUGENO (Baseado em Funções Matemáticas)
# Como o skfuzzy não tem uma classe de controle pronta para Sugeno, 
# a literatura do Python faz o import das funções de pertinência puras:
from skfuzzy import trimf, gaussmf  
# E a saída é calculada via combinação linear matemática (ex: usando NumPy)
import numpy as np