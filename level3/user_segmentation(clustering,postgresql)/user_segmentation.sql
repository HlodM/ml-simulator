WITH t1 AS (
  SELECT
    email_id,
    SUM(amount) AS tot_amount
  FROM
    new_payments
  WHERE
    mode IN ('MasterCard', 'Visa', 'МИР')
    AND status = 'Confirmed'
  GROUP BY
    email_id
)
SELECT
  CONCAT(lower_bound, '-', upper_bound) AS purchase_range,
  (
    SELECT
      COUNT(email_id)
    FROM
      t1
    WHERE
      tot_amount > lower_bound
      AND tot_amount <= upper_bound
  ) AS num_of_users
FROM
  (
    VALUES
      (0, 20000),
      (20000, 40000),
      (40000, 60000),
      (60000, 80000),
      (80000, 100000),
      (100000, (SELECT MAX(tot_amount) FROM t1)::int)
  ) AS t (lower_bound, upper_bound)