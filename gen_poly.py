import os
import sympy as sy
from sympy.abc import x
import numpy as np
import itertools as it
from sympy.parsing.sympy_parser import parse_expr
from collections import Counter
from functools import reduce
from collections import OrderedDict

import subprocess
import time

def expandMultipleFactor(factors):
    ''' Представляет полином f(x)**k в виде k полиномов f(x) '''
    ''' Например, (x**2 + 1)**3 => [(x**2 + 1), (x**2 + 1), (x**2 + 1)]'''
    ''' Для упрошения логики работы последующих функций'''
    result = []        
    for f in factors:
        fb = f.as_base_exp()
        for m in range(fb[1]):
            result.append(fb[0])   
    return result

def getFactorsFromMaxima(n):
    ''' Разложение полинома x**n + 1 на множители над полем GF(2) с 
    помощью программы Maxima, т.к. для тех n, которые дают коды БЧХ, 
    SymPy непозволительно медленно раскладывает на множители (31, 63, 127 ...)
    '''
    ns = str(n)
    cmd = r'maxima -r "factor(x^' \
        + ns + \
        r' + 1), modulus:2; stringout(\"factor.txt\", %o1);quit();"'
    with open(os.devnull, "w") as f:
        ret_code = subprocess.call(cmd, shell=True, stdout=f)    
    if ret_code:
        return []

    # Читаем из файла результат разложения, сохраненный программой Maxima
    with open("factor.txt", "r") as f:
        factor = f.read().strip();
    # Перевод из стиля Maxima в стиль SymPy, чистка от лишних знаков
    factor = factor.replace(";", "")
    factor = factor.replace("-x", "+x")
    factor = factor.replace("(+x^", "(x^")

    factors = factor.split('*')
    factors = [i.replace("^","**") for i in factors]
    factors = [parse_expr(i) for i in factors]
    return factors

def doFactors(n):
    factors = getFactorsFromMaxima(n)

    if not factors:
        print("Программа Maxima не найдена. Используем возможности SymPy...")
        factors = list(sy.factor(x ** n - 1, modulus = 2).args)
        print('\n')
        print(factors)
        print("Факторизация выполнена с помошью библиотеки SymPy\n")
        return expandMultipleFactor(factors)
    else:
        print('\n')
        print(factors)
        print("Факторизация выполнена в программе Maxima\n")

    return expandMultipleFactor(factors)

def findGenPolys(factors, r):
    candidates = set()
    for n_factors in range(1, len(factors) + 1):
        for combs in it.combinations(factors, n_factors):
            z = reduce(lambda x, y: sy.expand(x * y, modulus = 2), combs)
            candidates.add(z)  
    candidates = list(candidates)

    list_polynoms = []
    for poly in candidates:
        deg = sy.degree(poly, x)
        if deg == r:
            list_polynoms.append(poly)

    return list_polynoms

def getGenMatrix(pp, n, r):
    '''Возвращает порождающую матрицу G циклического кода'''
    g = list(map(int, sy.poly(pp).all_coeffs()))
    g = g[::-1] # реверс
    for k in range(n - r - 1):
        g.append(int(0))
    ga = np.array(g)
    G = ga.copy()
    for k in range(n - r - 1):
        G = np.append(G, np.roll(ga, k + 1))
    return G.reshape(n - r, n)

def getSpectrum(gen_poly, n, r):
    x_pows = list()
    x_pow_i = 1
    for i in range(1, n - r + 1):
        x_pows.append(x_pow_i) # 1, x, x^2, ... , x^k-1
        x_pow_i *= x
    code_polys = list()
    for i in range(1, n - r + 1):
        combs = list(it.combinations(x_pows, i))
        for comb in combs:
            pol = 0
            for term in comb:             
                pol = pol + term        
            code_polys.append(sy.expand(pol * gen_poly, modulus = 2))
    weight = list(str(s.count(1) + s.count(x)) for s in code_polys)
    sp = {'0': 1}
    _sp = Counter(weight)
    sp.update(_sp)
    result = OrderedDict()
    for key in sorted(sp, key = lambda i: int(i)):
        result.update({int(key): sp[key]})
    return sorted(result.items(), key = lambda x: x[0])

def findTheBestPoly(polys, n, r):
    '''Возвращает полином с максимальным кодовым растоянием'''
    n_polys = len(polys)
    spectrums = [getSpectrum(polys[x], n, r) for x in range(n_polys)]
    d_codes = [spectrums[x][1][0] for x in range(n_polys)]
    bestDistance = max(d_codes)
    position = d_codes.index(bestDistance)
    bestPoly = polys[position] 
    return bestPoly, spectrums[position], d_codes

def getCheckPoly(gen_poly, n):
    return sy.div(x ** n - 1, gen_poly, modulus = 2)[0]


if __name__== "__main__":
    print('Генератор циклических кодов')
    print('---------------------------')
    print('Введите n')
    n = int(input())
    print('Введите r')
    r = int(input())

    start_time = time.time()

    print('Факторизация нуль-полинома (x^{}+1) над полем GF(2)...'.format(n))

    factors = doFactors(n)
    gen_polys = findGenPolys(factors, r)
    if not gen_polys:
        print('Порождающего полинома степени {} не найдено!'.format(r))
        exit()
    else:
        print('Найденные порождающие полиномы степени {}'.format(r))
        print(gen_polys)
    good_poly, spectrum, d_codes = findTheBestPoly(gen_polys, n, r)
    bestDistance = max(d_codes)
    print('Список кодовых расстояний соответствующих циклических кодов')
    print(d_codes)
    print('Порождающий полином g(x), дающий наибольшее кодовое расстояние, равное {}'.format(bestDistance))
    print(good_poly)

    gen_matrix = getGenMatrix(good_poly, n, r)
    print('Порождающая матрица G кода ({}, {})'.format(n, n - r))
    print(gen_matrix)
    check_poly = getCheckPoly(good_poly, n)
    print('Проверочный полином h(x)')
    print(check_poly)
    print('Спектр кода (w, N), где w - вес кодового слова, N - количество таких слов')
    print(spectrum)

    print('Время счета {:.2f} минут'.format((time.time() - start_time)/60))

