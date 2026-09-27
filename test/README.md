# SurveyPilot 测试说明

`test/` 保存后端的单元测试、接口测试和少量可选的在线联调脚本。普通测试使用 Python 标准库 `unittest`：

```bash
uv run python -m unittest discover -s test -v
```

前端类型检查与打包使用：

```bash
npm run front:build
```

需要真实模型连接的测试默认跳过。只有在本地 `config/model.json` 已配置有效模型、并且明确要调用外部服务时，才运行对应的在线测试，例如：

```bash
RUN_LIVE_MODEL_TEST=1 uv run python -m unittest test.test_live_model_config -v
```

在线测试可能产生模型费用。`config/model.json`、`config/system.yaml` 以及运行数据均不提交到仓库。
