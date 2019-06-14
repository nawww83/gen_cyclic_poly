import os
import sympy as sy
from sympy.abc import x
import numpy as np
import itertools as it
from sympy.parsing.sympy_parser import parse_expr
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
    factor = factor.replace("-1", "+1")

    factors = factor.split('*')
    factors = [i.replace("^","**") for i in factors]
    factors = [parse_expr(i) for i in factors]
    return factors

def doFactors(n):
    factors = getFactorsFromMaxima(n)

    if not factors:
        print("Программа Maxima не найдена. Используем возможности SymPy...")
        factors = list(sy.factor(x ** n - 1, modulus = 2).args)
        for f in factors:
            print(f)
        print("Факторизация выполнена с помошью библиотеки SymPy\n")
        return expandMultipleFactor(factors)
    else:
        for f in factors:
            print(f)
        print("Факторизация выполнена в программе Maxima\n")

    return expandMultipleFactor(factors)

def composeGenPolys(factors, r):
    candidates = set()
    for n_factors in range(1, len(factors) + 1):
        for combs in it.combinations(factors, n_factors):
            degs = [p.as_poly().degree() for p in combs]
            if sum(degs) != r:
                continue
            else:
                z = reduce(lambda x, y: sy.expand(x * y, modulus = 2), combs)
                candidates.add(z)  
    return list(candidates)

def withoutMirrors(polys):
    ''' Фильтрует зеркальные полиномы '''
    result = []
    uniq = []
    for p in polys:
        cl = p.as_poly().all_coeffs()
        cl_mirror = cl[::-1]
        if cl not in uniq:
            uniq.append(cl)
            uniq.append(cl_mirror)
            result.append(p)
    return result

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
    ''' Возвращает спектр кода по его порождающему полиному g(x) '''
    ''' Расчет методом полного перебора информационных полиномов (медленно)'''
    zeros_ = [0 for _ in range(n - r)]
    g_x = gen_poly.as_poly(modulus = 2)
    powers_ = [i_ for i_ in range(n - r)] # [0, 1, 2, ..., k - 1]
    sp = dict()
    for i in range(n + 1):
        sp[i] = 0
    sp[0] = 1
    for i in range(1, n - r + 1):
        combs = list(it.combinations(powers_, i)) # i = 1...k
        for c_ in combs:
            a_ = zeros_.copy()
            for index in c_:
                a_[index] = 1
            a_x = sy.Poly.from_list(a_, x, modulus = 2)
            d_ = sy.expand(a_x * g_x)
            sp[d_.count(1) + d_.count(x)] += 1
    # удаление элементов с нулевым значением
    filtered = {k: v for k, v in sp.items() if v > 0}
    sp.clear()
    sp.update(filtered)
    # сортировка словаря по ключу
    result = OrderedDict()
    for key in sorted(sp, key = lambda i: i):
        result.update({key: sp[key]})
    return sorted(result.items(), key = lambda x: x[0])
    
def weightPoly(p):
    ''' Возвращает хеммингов вес полинома'''
    return p.count(1) + p.count(x)

def sortPolys(polys, desc = True):
    ''' Сортирует полиномы в порядке убывания их веса'''
    weights = [weightPoly(p) for p in polys]
    indexies = sorted(range(len(weights)), \
        key = weights.__getitem__, \
        reverse = desc)
    return [polys[i] for i in indexies]

def findTheBestPoly(polys, n, r):
    ''' Возвращает порождающий полином с максимальным кодовым растоянием '''
    ''' Полиномы должны быть отсортированы в порядке убывания их веса '''
    n_polys = len(polys)
    max_d = 0
    for p in polys:
        weight = weightPoly(p)
        sp = getSpectrum(p, n, r)
        d = sp[1][0]
        if d > max_d: # нашли лучший код
            bestPoly = p
            max_d = d
            good_sp = sp
        if weight <= max_d: # нет смысла искать лучший полином среди 
            # полиномов, вес которых меньшн текущего наибольшего расстояния кода
            break
    return bestPoly, good_sp, max_d

def getCheckPoly(gen_poly, n):
    ''' Возвращает проверочный полином h(x) '''
    return sy.div(x ** n - 1, gen_poly, modulus = 2)[0]


if __name__== "__main__":
    print('Генератор циклических (n, k)-кодов')
    print('---------------------------')
    print('Введите длину кода n')
    n = int(input())
    print('Введите число проверочных символов r')
    r = int(input())

    start_time = time.time()

    print('Факторизация нуль-полинома (x^{}+1) над полем GF(2)...'.format(n))

    factors = doFactors(n)
    print('Формирование порождающих полиномов степени r={}...'.format(r))
    gen_polys = composeGenPolys(factors, r)
    print('Фильтрация зеркальных полиномов...')
    gen_polys = withoutMirrors(gen_polys)
    print('Сортировка полиномов в порядке убывания их веса...')
    gen_polys = sortPolys(gen_polys)
    if not gen_polys:
        print('Порождающего полинома степени r={} не найдено!'.format(r))
        exit()
    else:
        print('Найденные порождающие полиномы')
        for p in gen_polys:
            print(p)
    print('Поиск наилучшего полинома по критерию максимального кодового расстояния...')
    good_poly, spectrum, bestDistance = findTheBestPoly(gen_polys, n, r)
    print('Наилучший порождающий полином g(x)')
    print(good_poly)
    print('Кодовое расстояние {}'.format(bestDistance))

    #gen_matrix = getGenMatrix(good_poly, n, r)
    #print('Порождающая матрица G кода ({}, {})'.format(n, n - r))
    #print(gen_matrix)
    check_poly = getCheckPoly(good_poly, n)
    print('Проверочный полином h(x)')
    print(check_poly)
    print('Спектр кода (w, N), где w - вес кодового слова, N - количество таких слов')
    print(spectrum)

    print('Время счета {:.2f} минут'.format((time.time() - start_time)/60))

