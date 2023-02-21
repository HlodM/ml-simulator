Запрос для сегментации пользователей на основе данных из таблицы по оплате. 
Учитываются только успешные оплаты пользователей, совершенные через банковскую карту ('MasterCard', 'МИР', 'Visa').

Запрос возвращает колонки purchase_range (с шагом 20 тыс до 100 тыс и 100 тыс +) и num_of_users (количество пользователей).

![alt text](https://github.com/HlodM/ml-simulator/blob/main/level3/user_segmentation(clustering%2Cpostgresql)/segmentation_bar.png)
