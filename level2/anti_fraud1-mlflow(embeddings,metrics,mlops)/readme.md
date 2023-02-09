Функция job выполняет и логирует расчеты метрик для пяти разных запусков и трех разных типов моделей 
(IsolationForest, OneClassSVM, LocalOutlierFactor). На основании сравнения результатов можно выбрать production модель.

В качестве метрик расчитываются:
- recall_at_precision - лучшее значение recall для значений precision выше заданного минимального порога.
- recall_at_specificity - аналогично, только вместо precision используется specificity.

Инструменты: sklearn, numpy, mlflow.