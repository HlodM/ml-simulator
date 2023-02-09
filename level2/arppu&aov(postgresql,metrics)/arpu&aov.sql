SELECT
  date_trunc('month', date) :: date AS time,
  SUM(amount) / COUNT(DISTINCT email_id) AS arppu,
  SUM(amount) / COUNT(*) AS aov
FROM
  new_payments
WHERE
  status = 'Confirmed'
GROUP BY
  time
ORDER BY
  time