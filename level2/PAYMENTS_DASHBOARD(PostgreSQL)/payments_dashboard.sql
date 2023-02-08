SELECT
  date_trunc('month', date) :: date AS time,
  mode,
  COUNT(status) FILTER(
    WHERE
      status = 'Confirmed'
  ) * 100./ COUNT(*) AS percents
FROM
  new_payments
WHERE
  mode != 'Не определено'
GROUP BY
  time,
  mode
ORDER BY
  time,
  mode