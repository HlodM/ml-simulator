SELECT
  user_id,
  item_id,
  SUM(units) AS qty,
  ROUND(price, 2) AS price
FROM
  (
    SELECT
      user_id,
      item_id,
      units,
      AVG(price) OVER (PARTITION BY item_id) AS price
    FROM
      default.karpov_express_orders
    WHERE
      timestamp BETWEEN %(start_date)s
      AND %(end_date)s
  ) AS t1
GROUP BY
  item_id,
  user_id,
  price
ORDER BY
  user_id,
  item_id