Запрос для формирования из таблицы активности пользователей следующих фич, сгруппированных по дням и user_id:
- days_offline – число дней начиная с последнего захода на сайт (0 – если в этот день были решения; 1 – если сегодня пользователь не заходил на сайт, но заходил вчера; 2 – если позавчера и т.д.)
- avg_submits_14d – среднее число попыток за последние 14 дней (включая текущий)
- success_rate_14d –  доля успешных попыток за последние 14 дней (включая текущий)
- solved_total  – суммарное число успешных сабмитов (к настоящему дню)
- target_14d – был ли пользователь онлайн следующие 14 дней? (1 – если не был, 0 – если был)

TODO: hard part
