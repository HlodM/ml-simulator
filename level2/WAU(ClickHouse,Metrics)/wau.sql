SELECT
  DISTINCT toDate(timestamp) AS day,
  COUNT(DISTINCT user_id) OVER (
    ORDER BY
      day RANGE BETWEEN 6 PRECEDING
      AND CURRENT ROW
  ) AS wau
FROM
  default.churn_submits