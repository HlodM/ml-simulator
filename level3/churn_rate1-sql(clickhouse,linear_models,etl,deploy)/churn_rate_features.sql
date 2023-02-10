WITH t1 as (
  SELECT
    user_id,
    toDate(timestamp) AS day,
    COUNT(submit) AS n_submits,
    COUNT(DISTINCT task_id) AS n_tasks,
    SUM(is_solved) AS n_solved
  FROM
    default.churn_submits
  GROUP BY
    user_id,
    day
),
t2 AS (
  SELECT
    user_id,
    day
  FROM
    (
      SELECT
        DISTINCT user_id
      FROM
        default.churn_submits
    ) AS tp1,
    (
      SELECT
        DISTINCT toDate(timestamp) day
      FROM
        default.churn_submits
    ) AS tp2
),
t3 AS (
  SELECT
    t2.user_id,
    t2.day,
    COALESCE(n_submits, 0) AS n_submits,
    COALESCE(n_tasks, 0) AS n_tasks,
    COALESCE(n_solved, 0) AS n_solved
  FROM
    t2
    LEFT JOIN t1 ON t2.user_id = t1.user_id
    AND t2.day = t1.day
  ORDER BY
    user_id,
    day
),
t4 AS (
  SELECT
    user_id,
    day,
    MAX(n_submits) OVER w3 AS max_follow_submit_14d,
    SUM(n_submits) OVER w1 AS sum_submits_14d,
    SUM(n_solved) OVER w1 AS sum_solved_14d,
    SUM(n_solved) OVER w2 AS solved_total,
    MAX(submit_day) OVER w2 AS last_submit_day
  FROM
    (
      SELECT
        tt1.user_id,
        tt1.day,
        tt1.n_submits,
        tt1.n_solved,
        tt2.day AS submit_day
      FROM
        t3 AS tt1
        LEFT JOIN (
          SELECT
            user_id,
            day
          FROM
            t3
          WHERE
            n_submits > 0
        ) tt2 ON tt1.user_id = tt2.user_id
        AND tt1.day = tt2.day
    ) AS t WINDOW w1 AS (
      PARTITION BY user_id
      ORDER BY
        day ROWS BETWEEN 13 PRECEDING
        AND CURRENT ROW
    ),
    w2 AS (
      PARTITION BY user_id
      ORDER BY
        day ROWS BETWEEN UNBOUNDED PRECEDING
        AND CURRENT ROW
    ),
    w3 AS (
      PARTITION BY user_id
      ORDER BY
        day ROWS BETWEEN 1 FOLLOWING
        AND 14 FOLLOWING
    )
)
SELECT
  user_id,
  day,
  day - last_submit_day AS days_offline,
  sum_submits_14d / 14 AS avg_submits_14d,
  sum_solved_14d / IF(sum_submits_14d = 0, 1, sum_submits_14d) AS success_rate_14d,
  solved_total,
  IF(max_follow_submit_14d > 0, 0, 1) AS target_14d
FROM
  t4 st SETTINGS join_use_nulls = 1