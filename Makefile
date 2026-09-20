VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: help setup lint typecheck test verify smoke-demo demo ml-report all clean

help: ## 显示全部命令
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## 安装（核心 + dev + ml extras，可编辑模式）
	$(PY) -m pip install -e ".[dev,ml]" || python -m pip install -e ".[dev,ml]"

lint: ## ruff check + format 检查
	ruff check . && ruff format --check .

typecheck: ## mypy（宽松档）
	mypy src

test: ## 单元测试
	pytest -q

verify: ## 外部真值物理验证（轨道/多普勒/链路预算/波形分辨率）
	python -m kimi_isac.verify

smoke-demo: ## 秒级闭环冒烟（合成数据，不训练）
	python -m kimi_isac.closedloop --smoke

demo: ## 感知-通信闭环 demo（权重在 results/ 下缓存）
	python -m kimi_isac.closedloop

ml-report: ## ML 感知报告（多种子 + bootstrap CI）
	python -m kimi_isac.ml.report

all: lint typecheck test verify ml-report demo ## 全量验收

clean: ## 清理运行产物
	rm -rf results/
