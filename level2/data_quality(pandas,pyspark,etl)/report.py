"""DQ Report."""

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field
from user_input.metrics import Metric
from joblib import hash as jhash

import pandas as pd
import pyspark.sql as ps


LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"
    memory_: Dict = field(default_factory=dict)

    def fit(self, tables: Dict[str, Union[pd.DataFrame, ps.DataFrame]]) -> Dict:
        """Calculate DQ metrics and build report."""

        if self.engine == "pandas":
            return self._fit_pandas(tables)

        if self.engine == "pyspark":
            return self._fit_pyspark(tables)

        raise NotImplementedError("Only pandas and pyspark APIs currently supported!")

    @staticmethod
    def _hash_pandas_dict(tables: Dict[str, pd.DataFrame]) -> str:
        """Returns hash of dictionary with pd.DataFrames"""
        return jhash({key: tables[key] for key in sorted(tables.keys())})

    @staticmethod
    def _hash_pyspark_dict(tables: Dict[str, ps.DataFrame]) -> str:
        """Returns hash of dictionary with ps.DataFrames"""
        return jhash({key: tables[key].collect() for key in sorted(tables.keys())})

    def _build_report(self, tables: Dict[str, Union[pd.DataFrame, ps.DataFrame]], report: Dict) -> None:
        """Calculate DQ metrics and build report"""
        data = []

        for (table_name, metric, limits) in self.checklist:
            try:
                value = metric(tables[table_name])
            except Exception as err:
                error = err
                value = {}
                status = "E"
            else:
                error = ""
                if limits:
                    key, (lower_bound, upper_bound) = next(iter(limits.items()))
                    status = "." if lower_bound <= value[key] <= upper_bound else "F"
                else:
                    status = "."

            data.append((table_name, repr(metric), repr(limits), value, status, error))

        columns = ["table_name", "metric", "limits", "values", "status", "error"]
        result = pd.DataFrame(data=data, columns=columns)

        report["result"] = result
        report["title"] = f"DQ Report for tables {sorted(tables.keys())}"
        report["total"] = len(report["result"])

        status_dict = {"passed": ".", "failed": "F", "errors": "E"}
        for key, value in status_dict.items():
            report[key] = sum(report["result"]["status"] == value)
            report[key+"_pct"] = round(100 * report[key] / report["total"], 2)

    def _fit_pandas(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: Pandas"""

        self.report_ = {}
        report = self.report_

        hash_tables = self._hash_pandas_dict(tables)

        if hash_tables not in self.memory_:
            self._build_report(tables, report)
            self.memory_[hash_tables] = report
        else:
            self.report_ = self.memory_[hash_tables]

        return report

    def _fit_pyspark(self, tables: Dict[str, ps.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: PySpark"""

        self.report_ = {}
        report = self.report_

        hash_tables = self._hash_pyspark_dict(tables)

        if hash_tables not in self.memory_:
            self._build_report(tables, report)
            self.memory_[hash_tables] = report
        else:
            self.report_ = self.memory_[hash_tables]

        return report

    def to_str(self) -> str:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before usong this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )
