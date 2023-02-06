# run_file_demo
from rqalpha import run_file

config = {
  "base": {
    "start_date": "2019-01-01",
    "end_date": "2019-12-31",
    "benchmark": "IF88",
    "accounts": {
        "future": 10000000
    }
  },
  "extra": {
    "log_level": "verbose",
  },
  "mod": {
    "sys_analyser": {
      "enabled": True,
      "plot": True
    }
  }
}

strategy_file_path = "./IF_DRL.py"

run_file(strategy_file_path, config)