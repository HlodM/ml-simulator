Функция dcg для вычисления метрики DCG(discounted cumulative gain) двумя методами:
- standatd - штраф релевантности на вес, равный логарифму номера позиции в выдаче.
- Industry – экспоненциально увеличивает релевантность.

Функция normalized_dcg аналлогично вычисляет нормированную метрику nDCG (Normalized Discounted Cumulative Gain) как отношение DCG к максимально возможному DCG для конкретного запроса.

Функция avg_ndcg вычисляет метрику Avarage nDCG - усредненное значение метрики nDCG по каждому запросу из множества. 